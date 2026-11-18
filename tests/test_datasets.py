"""Reading .npy / .nc / .h5 files into the 7-D layout (N, T, F, C, D, H, W)."""
import numpy as np
import pytest


def test_axes_layout_reorders_channels_last(workspace):
    from app.catalog import DatasetCard
    from app.datasets import DatasetHandle

    path = workspace["data"] / "chlast.npy"
    card = DatasetCard(id="a", name="a", source={"type": "local", "path": str(path)}, layout={"axes": "NTDHWF"})
    handle = DatasetHandle(card, path)
    assert handle.shape == (3, 5, 1, 1, 8, 8, 8)
    raw = np.load(path)
    X = handle.read([2, 0])  # rows come back in sorted order: 0, 2
    assert np.allclose(X[1, 3, 0, 0], raw[2, 3, ..., 0])


def test_size_layout_must_match_the_file(workspace):
    from app.catalog import DatasetCard
    from app.datasets import DatasetHandle

    path = workspace["data"] / "two-d.npy"
    card = DatasetCard(id="b", name="b", source={"type": "local", "path": str(path)},
                       layout={"fields": 3, "height": 16, "width": 16})
    with pytest.raises(ValueError, match="can't be reshaped"):
        DatasetHandle(card, path)


def test_netcdf_and_hdf5_read_only_requested_rows(tmp_path):
    import h5py
    import netCDF4

    from app.catalog import DatasetCard
    from app.datasets import DatasetHandle

    arr = np.arange(5 * 3 * 2 * 8, dtype=np.float32).reshape(5, 3, 2, 8)
    with netCDF4.Dataset(tmp_path / "d.nc", "w") as nc:
        for dim, size in zip("abcd", arr.shape):
            nc.createDimension(dim, size)
        nc.createVariable("x", "f4", ("c",))[:] = np.arange(2)       # small coordinate array: not picked
        nc.createVariable("solution", "f4", tuple("abcd"))[:] = arr
    with h5py.File(tmp_path / "d.h5", "w") as h5:
        h5.create_dataset("grp/u", data=arr)

    for name in ("d.nc", "d.h5"):
        card = DatasetCard(id="n", name="n", source={"type": "local", "path": str(tmp_path / name)},
                           layout={"axes": "NTFW"})
        handle = DatasetHandle(card, tmp_path / name)
        assert handle.var["name"] in ("solution", "grp/u")  # the largest array is used by default
        assert np.array_equal(handle.read([4])[0, :, :, 0, 0, 0, :], arr[4])


def test_inspect_lists_arrays_in_a_file(client, workspace):
    res = client.post("/api/inspect", json={"source": {"type": "local", "path": str(workspace["data"] / "chlast.npy")}})
    assert res.json()["variables"][0]["shape"] == [3, 5, 8, 8, 8, 1]
