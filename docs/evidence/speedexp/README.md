# speed-exp (2026-11-05, robot ~/chefbot_speedexp = e42fe35, cap 0.5 m/s, accel/brake 0.25; Jack on e-stop)
Stops (server frozen by a robot-side watcher; odometry / mocap; model tau 0.333, brake 0.25):
- exp35: 0.323 m/s -> 0.311 / 0.326 m, 1.5 s (model 0.316)
- exp50: 0.478 m/s -> 0.574 / 0.594 m, 2.1 s (model 0.616)
Parks seed 1 (`park --random 8 --seed 1 --map maps/src_counter --margin 0.09 --approach 0`):
- teleop 0.25: 114.4 s, dev max 55 mm, plate >= 92 mm
- exp50 0.5: 89.8 s (-22%), 8/8, err <= 1.8 mm / 0.39 deg, 0 stops, BUT dev max 118 mm and plate >= 33 mm (01_north 59, 03_west 33)
