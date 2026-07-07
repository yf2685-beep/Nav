"""Test which MP3D raw mesh (textured .obj vs vertex-colored .ply) loads AND renders in
habitat without segfault, via a stage config. Writes a sample RGB."""
import argparse, os, json, numpy as np, habitat_sim, magnum as mn, quaternion
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", default="/home/asus/Research/Nav/memnav_viz/mp3d_sample.png")
ap.add_argument("--lit", action="store_true", help="requires_lighting=True (else flat)")
args = ap.parse_args()

cfg = os.path.join(os.path.dirname(args.mesh),
                   os.path.splitext(os.path.basename(args.mesh))[0] + ".stage_config.json")
json.dump({"render_asset": os.path.basename(args.mesh), "up": [0, 0, 1], "front": [0, 1, 0],
           "requires_lighting": args.lit, "units_to_meters": 1.0}, open(cfg, "w"))

bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = cfg; bk.enable_physics = False
s = habitat_sim.CameraSensorSpec(); s.uuid = "c"; s.sensor_type = habitat_sim.SensorType.COLOR
s.resolution = [270, 480]; s.hfov = 68.0; s.position = mn.Vector3(0, 0, 0)
ac = habitat_sim.agent.AgentConfiguration(); ac.sensor_specifications = [s]
sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ac]))
ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = 0.3
sim.recompute_navmesh(sim.pathfinder, ns)
means = []
for k in range(6):
    p = sim.pathfinder.get_random_navigable_point()
    st = habitat_sim.agent.AgentState(); st.position = [p[0], p[1] + 0.5, p[2]]
    st.rotation = quaternion.from_rotation_vector([0, k, 0])
    sim.get_agent(0).set_state(st)
    img = sim.get_sensor_observations()["c"][..., :3]; means.append(float(img.mean()))
    if k == 0:
        Image.fromarray(img).save(args.out)
print("RESULT_OK nav_area=%.1f mean_brightness=%.1f saved=%s"
      % (sim.pathfinder.navigable_area, np.mean(means), args.out))
