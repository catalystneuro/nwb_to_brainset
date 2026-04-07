import datetime
import logging
from typing import Union, Dict, Any, Optional
import h5py
import pynwb
import pynwb.epoch
import pynwb.misc
import numpy as np

from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import Sex, Species


# ---------------------------------------------------------------------------
# Lazy-loading timeseries subclasses
# ---------------------------------------------------------------------------


class _LazyH5Mixin:
    """Mixin that allows h5py.Dataset to be stored in ArrayDict-based objects.

    On attribute access, h5py.Dataset is transparently converted to a numpy array.
    The NWB IO must remain open until all data is consumed (e.g. written to HDF5).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if not name.startswith("_") and isinstance(value, h5py.Dataset):
            # Replicate the size-consistency check from ArrayDict.__setattr__
            first_dim = self._maybe_first_dim()
            if first_dim is not None and value.shape[0] != first_dim:
                raise ValueError(
                    f"First-dimension mismatch: {name} has {value.shape[0]}, "
                    f"existing attributes have {first_dim}."
                )
            object.__setattr__(self, name, value)  # bypass numpy-only check
        else:
            super().__setattr__(name, value)  # normal path for numpy / private

    def __getattribute__(self, name):
        # Load h5py.Dataset → numpy transparently on first access
        if name not in ("__dict__", "keys"):
            if name in self.keys():
                val = self.__dict__[name]
                if isinstance(val, h5py.Dataset):
                    return val[:]
        return object.__getattribute__(self, name)

    def to_hdf5(self, file):
        """Write to HDF5, using the base class name to avoid the LazyLazy bug.

        IrregularTimeSeries.to_hdf5 writes self.__class__.__name__ as the
        object type. For LazyIrregularTimeSeries this becomes
        "LazyIrregularTimeSeries", and Data.from_hdf5 prepends another "Lazy"
        → KeyError. Fix: write the grandparent class name (the real temporaldata type).
        """
        super().to_hdf5(file)
        # Patch the object attr to the base temporaldata class name
        # MRO: LazyIrregularTimeSeries → _LazyH5Mixin → IrregularTimeSeries → ...
        base_name = type(self).__mro__[2].__name__
        file.attrs["object"] = base_name


class LazyIrregularTimeSeries(_LazyH5Mixin, IrregularTimeSeries):
    """IrregularTimeSeries with lazy h5py.Dataset support for data fields."""

    pass


class LazyRegularTimeSeries(_LazyH5Mixin, RegularTimeSeries):
    """RegularTimeSeries with lazy h5py.Dataset support for data fields."""

    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_scalar_column(series):
    """Return True if a pandas Series contains only plain scalars (safe for ArrayDict)."""
    if series.dtype.kind in ("i", "u", "f", "b"):
        return True
    if series.dtype.kind in ("S", "U", "O"):
        return all(
            isinstance(v, (str, bytes, int, float, bool, type(None)))
            for v in series.iloc[:5]
        )
    return False


# ---------------------------------------------------------------------------
# NWB inspection helper
# ---------------------------------------------------------------------------


def _fmt_cols(cols, max_show=8):
    shown = list(cols[:max_show])
    rest = len(cols) - max_show
    s = ", ".join(shown)
    return s + f"  + {rest} more" if rest > 0 else s


def _key_to_nwb_path(key):
    """Convert a __-separated key to an NWB accessor path for display."""
    if key == "units":
        return "nwbfile.units"
    if key == "trials":
        return "nwbfile.trials"
    parts = key.split("__")
    if parts[0] == "intervals":
        return f"nwbfile.intervals['{parts[1]}']"
    if parts[0] == "acquisition":
        return f"nwbfile.acquisition['{parts[1]}']"
    if parts[0] == "processing":
        return f"nwbfile.processing['{parts[1]}']['{parts[2]}']"
    return key


# ---------------------------------------------------------------------------
# Nwb2Brainset class
# ---------------------------------------------------------------------------


class Nwb2Brainset:
    """Converts an NWB file to temporaldata objects for use in torch_brain.

    Subclass and override any ``convert_*`` method to customise how a specific
    modality is converted, without touching the rest of the pipeline.

    Parameters
    ----------
    nwbfile : pynwb.file.NWBFile
        An already-opened NWB file object.
    lazy_loading : bool
        If True, data arrays are kept as h5py.Dataset references until first
        access (e.g. write time). Timestamps are always loaded eagerly. The
        NWB IO must stay open until data is consumed.

    Examples
    --------
    Direct instantiation (user manages IO):

    >>> from pynwb import NWBHDF5IO
    >>> io = NWBHDF5IO("recording.nwb", "r")
    >>> converter = Nwb2Brainset(io.read())
    >>> converter.check()
    >>> data = converter.load(include=["units", "acquisition__LFP"])
    >>> io.close()

    From file with automatic cleanup:

    >>> with Nwb2Brainset.from_file("recording.nwb") as converter:
    ...     data = converter.load()

    Naming scheme for include/exclude keys
    ---------------------------------------
        units, spikes                  — from nwbfile.units
        trials                         — from nwbfile.trials
        intervals__<name>              — from nwbfile.intervals[<name>]
        acquisition__<name>            — from nwbfile.acquisition[<name>]
        processing__<mod>__<iface>     — from nwbfile.processing[<mod>][<iface>]
    """

    # --- Dispatch tables ---
    # Name-based: maps specific object keys to converter methods or callables.
    # Takes priority over type-based dispatch.
    converters_by_name = {}

    # Type-based: maps pynwb types to converter method names.
    # Order matters: first isinstance match wins.
    converters_by_type = {
        pynwb.misc.Units: "convert_units",
        pynwb.TimeSeries: "convert_timeseries",
        pynwb.epoch.TimeIntervals: "convert_interval",
    }

    def __init__(
        self,
        nwbfile,
        lazy_loading: bool = True,
        brainset_description: Optional[dict] = None,
    ):
        self.nwbfile = nwbfile
        self.lazy_loading = lazy_loading
        self.brainset_description = brainset_description or {}
        self.include = None
        self.exclude = None
        self.data = None
        self._io = None

    # --- File opening ---

    @classmethod
    def from_file(cls, path: str, lazy_loading: bool = True, **kwargs):
        """Open a local NWB file and return a converter instance.

        Use as a context manager to ensure the file is closed after conversion:

        >>> with Nwb2Brainset.from_file("recording.nwb") as converter:
        ...     data = converter.load()
        """
        from pynwb import NWBHDF5IO

        io = NWBHDF5IO(path, "r")
        instance = cls(io.read(), lazy_loading=lazy_loading, **kwargs)
        instance._io = io
        return instance

    # --- Lifecycle ---

    def close(self):
        """Close the underlying NWB IO, if opened via ``from_file``."""
        if self._io is not None:
            self._io.close()
            self._io = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Filtering ---

    def wanted(self, key: str) -> bool:
        """Return True if key should be converted, given self.include/self.exclude."""
        if self.include is not None:
            return key in self.include
        if self.exclude is not None:
            return key not in self.exclude
        return True

    # --- Object discovery ---

    def _collect_objects(self, obj, prefix, objects):
        """Recurse into obj; if it matches the dispatch table, store it."""
        if self.dispatch(obj) is not None:
            objects[prefix] = obj
        else:
            for child in obj.children:
                if isinstance(child, pynwb.core.NWBDataInterface):
                    self._collect_objects(child, f"{prefix}__{child.name}", objects)

    def find_objects(self) -> dict:
        """Walk the NWB file and return all leaf neurodata objects as {path: obj}."""
        objects = {}
        if self.nwbfile.units is not None:
            objects["units"] = self.nwbfile.units
        if self.nwbfile.trials is not None:
            objects["trials"] = self.nwbfile.trials
        for name, obj in (self.nwbfile.intervals or {}).items():
            if name != "trials":
                objects[f"intervals__{name}"] = obj
        for name, obj in (self.nwbfile.acquisition or {}).items():
            self._collect_objects(obj, f"acquisition__{name}", objects)
        for mod_name, module in (self.nwbfile.processing or {}).items():
            for iface_name, iface in module.data_interfaces.items():
                self._collect_objects(
                    iface, f"processing__{mod_name}__{iface_name}", objects
                )
        return objects

    # --- Metadata extraction (overridable) ---

    def extract_brainset_description(self) -> Optional[BrainsetDescription]:
        """Create a BrainsetDescription. Users can pass ``brainset_description={"id": "my_dataset", "source": "..."}``
        to the constructor to override any default field.
        """
        defaults = dict(
            id="unknown",
            origin_version="0.0.0",
            derived_version="0.0.0",
            source="",
            description="",
        )
        defaults.update(self.brainset_description)
        return BrainsetDescription(**defaults)

    def extract_subject(self) -> Optional[SubjectDescription]:
        """Extract subject metadata from the NWB file.

        Reads species, sex, and subject_id from ``nwbfile.subject`` when available.
        Override this method to customise subject extraction.
        """
        subj = self.nwbfile.subject
        if subj is None:
            return None

        # Parse species
        species = Species.UNKNOWN
        if getattr(subj, "species", None):
            species_str = subj.species
            try:
                species = Species.from_string(species_str)
            except (ValueError, KeyError):
                logging.warning(f"Unknown species '{subj.species}' — using UNKNOWN")

        # Parse sex
        sex = Sex.UNKNOWN
        if getattr(subj, "sex", None):
            try:
                sex = Sex.from_string(subj.sex)
            except (ValueError, KeyError):
                logging.warning(f"Unknown sex '{subj.sex}' — using UNKNOWN")

        subject_id = getattr(subj, "subject_id", None) or "unknown"

        return SubjectDescription(
            id=subject_id.lower(),
            species=species,
            sex=sex,
        )

    def extract_session(self) -> Optional[SessionDescription]:
        """Extract session metadata from the NWB file.

        Uses ``session_id`` and ``session_start_time`` from the NWB file.
        Override this method to customise session extraction.
        """
        session_id = getattr(self.nwbfile, "session_id", None) or "unknown"
        recording_date = getattr(self.nwbfile, "session_start_time", None)
        if recording_date is None:
            recording_date = datetime.datetime.now()

        return SessionDescription(
            id=session_id,
            recording_date=recording_date,
        )

    # --- Overridable specific conversion methods ---

    def convert_units(self, obj):
        """Yield ("units", ArrayDict) and ("spikes", IrregularTimeSeries).

        Follows the IBL pipeline convention:
        - spike_times_index[:] returns a list of spike-train arrays (one per unit)
        - unit_index is sequential 0-indexed int64 (position in units.id array)
        - units.id is a string array ["unit_0", "unit_1", ...]
        - original NWB unit IDs are preserved as units.nwb_id
        """
        spike_train_list = obj.spike_times_index[:]
        n_units = len(spike_train_list)

        unit_ids = np.array([f"unit_{i}" for i in range(n_units)])

        spike_timestamps_list = []
        spike_unit_index_list = []
        for i, spike_train in enumerate(spike_train_list):
            spike_train = np.asarray(spike_train)
            spike_timestamps_list.append(spike_train)
            spike_unit_index_list.append(np.full(len(spike_train), i, dtype=np.int64))

        spike_timestamps = (
            np.concatenate(spike_timestamps_list)
            if n_units
            else np.array([], dtype=np.float64)
        )
        spike_unit_index = (
            np.concatenate(spike_unit_index_list)
            if n_units
            else np.array([], dtype=np.int64)
        )

        if len(spike_timestamps) > 0:
            spikes = IrregularTimeSeries(
                timestamps=spike_timestamps,
                unit_index=spike_unit_index,
                domain="auto",
            )
            spikes.sort()
        else:
            spikes = None

        _SKIP = {
            "spike_times",
            "spike_times_index",
            "obs_intervals",
            "obs_intervals_index",
            "electrodes",
            "electrodes_index",
            "waveform_mean",
            "waveform_sd",
        }
        metainfo = {
            "id": unit_ids,
            "nwb_id": np.asarray(obj.id[:]),
        }
        for col in obj.colnames:
            if col in _SKIP:
                continue
            try:
                raw = obj[col][:]
                arr = np.asarray(raw)
                if arr.dtype == object and arr.ndim == 1 and len(arr) > 0:
                    if isinstance(arr[0], (list, np.ndarray)):
                        logging.debug(f"Skipping ragged units column '{col}'")
                        continue
                if arr.ndim != 1 or len(arr) != n_units:
                    continue
                if arr.dtype.kind in ("S", "O", "U"):
                    arr = np.array(
                        [v.decode() if isinstance(v, bytes) else str(v) for v in arr]
                    )
                else:
                    arr = arr.astype(np.float32)
                metainfo[col] = arr
            except Exception as e:
                logging.debug(f"Skipping units column '{col}': {e}")

        yield "units", ArrayDict(**metainfo)
        if spikes is not None:
            yield "spikes", spikes

    def convert_timeseries(self, obj):
        """Convert a pynwb.TimeSeries → IrregularTimeSeries or RegularTimeSeries.

        When ``lazy_loading=True``, data is kept as an h5py.Dataset reference and
        only loaded into memory on first attribute access (e.g. at write time).
        Timestamps are always loaded eagerly.
        """
        d = obj.data if self.lazy_loading else obj.data[:]

        if isinstance(obj, pynwb.behavior.SpatialSeries):
            field_name = "position"
        else:
            field_name = "data"

        ts_cls_irreg = (
            LazyIrregularTimeSeries if self.lazy_loading else IrregularTimeSeries
        )
        ts_cls_reg = LazyRegularTimeSeries if self.lazy_loading else RegularTimeSeries

        if obj.timestamps is not None:
            t = np.asarray(obj.timestamps)
            return ts_cls_irreg(timestamps=t, **{field_name: d}, domain="auto")

        rate = getattr(obj, "rate", None)
        if rate is not None:
            domain_start = obj.starting_time if obj.starting_time is not None else 0.0
            n_samples = len(obj.data)
            domain_end = domain_start + (n_samples - 1) / rate
            return ts_cls_reg(
                **{field_name: d},
                sampling_rate=rate,
                domain=Interval(
                    start=np.array([domain_start]),
                    end=np.array([domain_end]),
                ),
            )

        logging.warning(
            f"TimeSeries '{obj.name}' has neither timestamps nor rate — skipping"
        )
        return None

    def convert_interval(self, obj):
        """Convert NWB TimeIntervals → Interval.

        Renames start_time/stop_time to start/end, pre-filters non-scalar columns
        (e.g. NWB object references, lists) so temporaldata never sees them, and
        passes any column ending in "_time" as a timekey.
        """
        if not hasattr(obj, "to_dataframe"):
            return None
        df = obj.to_dataframe()

        if "start_time" in df.columns and "stop_time" in df.columns:
            df = df.rename(columns={"start_time": "start", "stop_time": "end"})

        if "start" not in df.columns or "end" not in df.columns:
            logging.warning(f"{obj.name}: no start/end columns found — skipping")
            return None

        # Drop any column whose values aren't plain scalars (object refs, lists, etc.)
        scalar_cols = [c for c in df.columns if _is_scalar_column(df[c])]
        df = df[scalar_cols]

        time_cols = ["start", "end"] + [c for c in df.columns if c.endswith("_time")]
        try:
            interval = Interval.from_dataframe(df, timekeys=time_cols)
            if not interval.is_sorted():
                interval.sort()
            return interval
        except Exception as e:
            logging.warning(f"Failed to build Interval from {obj.name}: {e}")
            return None

    def dispatch(self, obj, key=None):
        """Return the converter method for a given pynwb object, or None.

        Checks ``converters_by_name`` first (keyed on the object's ``__``-separated
        path), then falls back to ``converters_by_type`` (keyed on pynwb type).
        Values can be method name strings or callables.
        """
        if key and key in self.converters_by_name:
            fn = self.converters_by_name[key]
            return getattr(self, fn) if isinstance(fn, str) else fn
        for obj_type, method_name in self.converters_by_type.items():
            if isinstance(obj, obj_type):
                return getattr(self, method_name)
        return None

    # --- Pipeline ---

    def load(
        self,
        include: Optional[list] = None,
        exclude: Optional[list] = None,
        auto_data: bool = True,
    ) -> Union[Data, Dict[str, Any]]:
        """Convert the NWB file to a Data object or plain dict.

        Discovers all leaf objects via ``find_objects``, then dispatches each
        to the appropriate converter via the ``converters`` table.

        Parameters
        ----------
        include : list[str], optional
            Only convert objects whose keys are in this list. Mutually exclusive
            with ``exclude``.
        exclude : list[str], optional
            Skip objects whose keys are in this list. Mutually exclusive with
            ``include``.
        auto_data : bool
            If True (default), wrap all converted objects in a ``Data`` container
            with automatic domain inference. If False, return a plain dict.
        """
        if include is not None and exclude is not None:
            raise ValueError("'include' and 'exclude' cannot be used together.")

        self.include = include
        self.exclude = exclude

        all_objects = self.find_objects()
        converted = {}

        for key, obj in all_objects.items():
            if not self.wanted(key):
                continue

            converter = self.dispatch(obj, key=key)
            if converter is None:
                logging.warning(
                    f"No converter for '{key}' ({type(obj).__name__}) — skipping"
                )
                continue

            try:
                result = converter(obj)
            except Exception as e:
                logging.warning(f"Failed to convert '{key}': {e}")
                continue

            # convert_units yields multiple items
            if hasattr(result, "__next__"):
                for sub_key, sub_val in result:
                    converted[sub_key] = sub_val
            elif result is not None:
                converted[key] = result

        if auto_data:
            # Inject metadata descriptions
            brainset_desc = self.extract_brainset_description()
            if brainset_desc is not None:
                converted["brainset"] = brainset_desc
            subject_desc = self.extract_subject()
            if subject_desc is not None:
                converted["subject"] = subject_desc
            session_desc = self.extract_session()
            if session_desc is not None:
                converted["session"] = session_desc

            self.data = Data(**converted, domain="auto")
            return self.data
        else:
            self.data = None
            return converted

    def save(self, path: str, **load_kwargs) -> None:
        """Convert (if needed) and save to an HDF5 file for torch_brain.

        If ``load()`` has already been called, reuses ``self.data``. Otherwise
        calls ``load(**load_kwargs)`` first.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``"session.h5"``).
        **load_kwargs
            Passed to ``load()`` if data hasn't been loaded yet
            (e.g. ``include``, ``exclude``).
        """
        if self.data is None:
            self.load(**load_kwargs)

        with h5py.File(path, "w") as f:
            self.data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
        logging.info(f"Saved to {path}")

    def check(self) -> list:
        """Print a formatted summary and return structured inspection data.

        Keys shown match the naming scheme used by load(include=..., exclude=...).

        Returns
        -------
        list[dict]
            Each dict contains ``key``, ``neurodata_type``, ``target_type``,
            plus type-specific fields (``n_units``, ``columns``, ``n_rows``,
            ``shape``, ``unit``, ``description``, ``sampling_rate``,
            ``duration``).
        """
        SEP = "=" * 65
        objects = self.find_objects()
        results = []

        for key, obj in objects.items():
            try:
                if isinstance(obj, pynwb.misc.Units):
                    n_units = len(obj)
                    cols = [c for c in obj.colnames if c != "spike_times"]
                    results.append(
                        {
                            "key": key,
                            "nwb_path": _key_to_nwb_path(key),
                            "neurodata_type": "Units",
                            "target_type": "ArrayDict",
                            "n_units": n_units,
                            "columns": cols,
                        }
                    )
                    results.append(
                        {
                            "key": "spikes",
                            "nwb_path": _key_to_nwb_path(key),
                            "neurodata_type": "Units",
                            "target_type": "IrregularTimeSeries",
                            "n_units": n_units,
                        }
                    )

                elif isinstance(obj, pynwb.epoch.TimeIntervals):
                    results.append(
                        {
                            "key": key,
                            "nwb_path": _key_to_nwb_path(key),
                            "neurodata_type": "TimeIntervals",
                            "target_type": "Interval",
                            "n_rows": len(obj),
                            "columns": list(obj.colnames),
                        }
                    )

                elif isinstance(obj, pynwb.TimeSeries):
                    neurodata_type = type(obj).__name__
                    shape = obj.data.shape if hasattr(obj.data, "shape") else ("?",)
                    unit = getattr(obj, "unit", None) or "?"
                    desc = (getattr(obj, "description", "") or "").strip()
                    generic = {"no description", "no comments", ""}
                    desc_str = desc[:60] if desc.lower() not in generic else ""
                    entry = {
                        "key": key,
                        "nwb_path": _key_to_nwb_path(key),
                        "neurodata_type": neurodata_type,
                        "shape": shape,
                        "unit": unit,
                        "description": desc_str,
                    }

                    rate = getattr(obj, "rate", None)
                    if rate is not None:
                        entry["target_type"] = "RegularTimeSeries"
                        entry["sampling_rate"] = rate
                    elif obj.timestamps is not None:
                        entry["target_type"] = "IrregularTimeSeries"
                        t0, t1 = float(obj.timestamps[0]), float(obj.timestamps[-1])
                        entry["duration"] = t1 - t0
                    else:
                        continue
                    results.append(entry)

            except Exception as e:
                logging.warning(f"check: could not inspect '{key}': {e}")

        # --- Print ---
        def _entries(target_type):
            return [r for r in results if r["target_type"] == target_type]

        print(SEP)
        print(" NWB File Contents")
        print(SEP)

        spiking = [
            r
            for r in results
            if r["neurodata_type"] == "Units" and r["target_type"] == "ArrayDict"
        ]
        print("\n[Spiking Data]")
        if spiking:
            for s in spiking:
                print(f"  {s['key']}  ({_key_to_nwb_path(s['key'])})")
                print(f"    n_units  : {s['n_units']:,}")
                if s.get("columns"):
                    print(f"    columns  : {_fmt_cols(s['columns'])}")
        else:
            print("  (none found)")

        intervals = _entries("Interval")
        print("\n[TimeIntervals]")
        if intervals:
            for iv in intervals:
                print(f"  {iv['key']}  ({_key_to_nwb_path(iv['key'])})")
                print(f"    n_rows   : {iv['n_rows']:,}")
                if iv.get("columns"):
                    print(f"    columns  : {_fmt_cols(iv['columns'])}")
        else:
            print("  (none found)")

        regular = _entries("RegularTimeSeries")
        print("\n[Regular TimeSeries]")
        if regular:
            for ts in regular:
                print(f"  {ts['key']}  ({_key_to_nwb_path(ts['key'])})")
                print(f"    rate     : {ts['sampling_rate']} Hz")
                print(f"    shape    : {ts['shape']}")
                print(f"    unit     : {ts['unit']}")
                if ts.get("description"):
                    print(f"    desc     : {ts['description']}")
        else:
            print("  (none found)")

        irregular = [r for r in _entries("IrregularTimeSeries") if "duration" in r]
        print("\n[Irregular TimeSeries]")
        if irregular:
            for ts in irregular:
                print(f"  {ts['key']}  ({_key_to_nwb_path(ts['key'])})")
                print(f"    duration : {ts['duration']:.2f} s")
                print(f"    shape    : {ts['shape']}")
                print(f"    unit     : {ts['unit']}")
                if ts.get("description"):
                    print(f"    desc     : {ts['description']}")
        else:
            print("  (none found)")

        print(f"\n{SEP}")

        return results
