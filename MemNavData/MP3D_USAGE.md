# mp3d_revisit_v0 data usage

Migration target:

- Root: `/home/chatsign/data/data-002/Nav/data/generated/mp3d_revisit_v0`
- Trajectories: `/home/chatsign/data/data-002/Nav/data/generated/mp3d_revisit_v0/vln_n1/traj_data`

## Included trajectory data

The migrated trajectory data contains only the MP3D revisit generations from:

- Local source: `/home/asus/Research/Nav/memnav_viz/mp3d_gen`
- Local source: `/home/asus/Research/Nav/memnav_viz/mp3d_2leg`
- Local source: `/home/asus/Research/Nav/memnav_viz/mp3d_3leg`

Remote layout:

- `vln_n1/traj_data/mp3d_gen/`
- `vln_n1/traj_data/mp3d_2leg/`
- `vln_n1/traj_data/mp3d_3leg/`
- `scripts/` contains the local `MemNavData` Python and markdown files used to
  download, generate, inspect, validate, compare, and visualize the data.

Each episode keeps the generated InternData-N1-style layout:

- `data/chunk-000/episode_000000.parquet`
- `videos/chunk-000/observation.images.rgb/*.jpg`
- `videos/chunk-000/observation.images.depth/*.png`
- `meta/gen_meta.json`
- `goal_image.jpg` and, when present, `goal_*.jpg`

## MP3D scenes used

Only one Matterport3D / MP3D scene is used in this migrated dataset:

| MP3D scene id | Scene file | Local asset files |
|---|---|---|
| `17DRP5sb8fy` | `17DRP5sb8fy.glb` | `/home/asus/Research/datasets/mp3d/17DRP5sb8fy.glb`, `/home/asus/Research/datasets/mp3d/17DRP5sb8fy.navmesh` |

All included generated episodes have `scene: "17DRP5sb8fy.glb"` in `meta/gen_meta.json`.

## Included episodes

| Target subdir | Episodes | Frames | Legs | Switch indices |
|---|---:|---:|---|---|
| `mp3d_gen/episode_0000` | 1 | 430 | 2 | `[233]` |
| `mp3d_gen/episode_0001` | 1 | 278 | 2 | `[142]` |
| `mp3d_gen/episode_0002` | 1 | 262 | 2 | `[151]` |
| `mp3d_2leg/episode_0000` | 1 | 293 | 2 | `[146]` |
| `mp3d_2leg/episode_0001` | 1 | 339 | 2 | `[171]` |
| `mp3d_3leg/episode_0000` | 1 | 506 | 3 | `[164, 324]` |
| `mp3d_3leg/episode_0001` | 1 | 423 | 3 | `[120, 274]` |

Totals:

- 7 generated episodes
- 2531 trajectory frames
- 1 MP3D scene id: `17DRP5sb8fy`

## Excluded local data

The local `../memnav_viz/twoleg_pp`, `../memnav_viz/twoleg_proto`, and
`../memnav_viz/twoleg_smooth` directories are not migrated into this dataset. Their
metadata uses `scene: "apartment_1.glb"`, which is the Habitat test apartment scene,
not MP3D / Matterport3D.

The local `../memnav_viz/intern` visual checks are also not trajectory data for this
MP3D revisit dataset.

## QA and manifest files

QA visualizations are copied under:

- `/home/chatsign/data/data-002/Nav/data/generated/mp3d_revisit_v0/qa/`

The structured manifest is copied to:

- `/home/chatsign/data/data-002/Nav/data/generated/mp3d_revisit_v0/manifests/mp3d_revisit_v0_manifest.json`

Project scripts and notes are copied to:

- `/home/chatsign/data/data-002/Nav/data/generated/mp3d_revisit_v0/scripts/`
