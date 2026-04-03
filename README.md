# nwb_to_brainset

Convert NWB files to [temporaldata](https://github.com/neuro-galaxy/temporaldata) and [brainsets](https://github.com/neuro-galaxy/brainsets) objects for use with [torch_brain](https://github.com/neuro-galaxy/torch_brain).

## What it does

`Nwb2Brainset` automatically discovers and converts all convertible objects in an NWB file:
- **Spiking data** → `ArrayDict` (units) + `IrregularTimeSeries` (spikes)
- **TimeSeries** → `IrregularTimeSeries` or `RegularTimeSeries`
- **TimeIntervals** → `Interval`

Supports lazy loading, `include`/`exclude` filtering, and is extensible via subclassing or per-key converter overrides.

## Quick start

```python
from load_nwb import Nwb2Brainset

with Nwb2Brainset.from_file("recording.nwb") as converter:
    converter.check()             # inspect available objects
    data = converter.load()       # convert to temporaldata.Data
    converter.save("session.h5")  # save for torch_brain
```

## Tutorial

See [load_nwb_tutorial.ipynb](load_nwb_tutorial.ipynb) for a full walkthrough, including selective loading, custom converters, and creating a torch_brain `Dataset` with trial-based train/valid/test splits.
