# Training storage policy

Check the destination drive immediately before **every** new data batch or episode and periodically during capture/export. Keep at least **10 GiB free**, in addition to the estimated next writes. Refuse a job if that space is unavailable. Do not delete existing recordings or checkpoints automatically.

`simulation_lab.storage.require_space` checks the actual filesystem on each call. The matched collector and robustness collector already use it. Browser recording now checks before starting and every 200 written actions; image export checks its estimated total raw RGB size before starting and rechecks every 30 images. The original pilot collector also checks before collection and saved trials. New command-state capture and HD video rendering use the same reserve.

Image export failures remain explicitly marked as failed and ineligible for training; recorded trajectories remain available. Filesystem checks cannot reserve space against unrelated applications, so bounded write intervals provide additional protection while a job runs.

Prefer compact state recordings and offline video generation over duplicating image datasets. Keep policy images at their existing training resolution; a higher-resolution demonstration video does not require a second training dataset.

Pilot/Colab ZIP packaging also checks the estimated full archive before starting and rechecks each input file. Existing archives are not overwritten. Held-out reset-state preparation checks before the batch and each case. These checks apply to evaluation copies and exports as well as newly collected training episodes.
