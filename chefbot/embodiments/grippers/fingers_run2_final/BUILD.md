# Printing and assembly — 54 mm full-entry cradle finger

This is a research prototype for guarded tests with cool, dry fixtures. No physical print or qualification has been performed. Food-contact, pad retention, fatigue, wet friction and real-controller release remain unqualified. Failed simulated trajectories must not be replayed on the robot.

## Parts and process

Print two identical registered body/pad assemblies on a **Bambu H2C**. Use **Bambu PETG HF** for `parts/body.stl` and **Bambu TPU 90A** for `parts/pad.stl`. The pad STL contains three connected components; keep their registered positions. The second complete finger mounts rotated180 degrees about the Hand vertical axis. No coating or hidden infill change is assumed.

Bambu's current TPU85/90 product guide explicitly supports TPU90A with its PETG series on H2C using mechanical interlocking. The earlier general H2C manual's high-temperature-filament example uses95A HF; this is a different, lower-temperature supported pairing. Use the right regular0.4mm hotend for TPU, external dry/top feed or the manufacturer's specified AMS HT bypass path; never ordinary AMS switching. PETG uses the left nozzle.

Apply the single rigid transform in `print_orientation.json` to the entire assembly. The build direction in finger coordinates is(1,1,0)/sqrt(2), with finger length in the layer plane. Drop the assembly as a whole, never its components individually. First TPU is approximately2.10mm above the lowest PETG backing edge before supports/raft. Place centrally in the shared dual-nozzle work area.

Starting profile to qualify:0.20mm layers, six walls, six top/bottom layers,100% infill in both materials; PETG250C, TPU225C, PETG bed65C, cooling/ventilated chamber25–45C. Keep the TPU preset's volumetric-speed limit and flow-dynamics calibration off;30–34mm/s is a conservative starting external-feature speed. TPU supplier bed30–35C applies to direct TPU deposition: here TPU begins on elevated PETG structure/supports. This elevated-bed combination still requires a thermal/adhesion coupon and sliced-layer review. PETG dry65C8h; TPU dry70C8h in a suitable forced-air filament oven, then feed/store below20%RH. Do not anneal the assembled finger.

Use a stable PETG support base and approximately8mm brim for the angled edge. Paint accessible breakaway supports under first leaf edges, the rear recess and backing overhangs; start around0.3mm interface separation and qualify removal. TPU90's stated unsupported overhang is about40degrees, so do not assume every45degree cavity wall will print unsupported. All2.4mm rear gaps open at lateral/distal edges. Inspect the slicer for islands and trapped supports. Reject a process that bonds support into a gap or tears a leaf. Print a representative key/leaf/support coupon before the pair.

## Geometry that must be preserved

The54mm finger spans52mm laterally, with a6mm clamp web and10mm rear root. The contact backing is5.6mm thick at x−2.5..3.1. All outer contact profiles are shifted6mm outward from the pocketed version: upper lips reach x15.5,z16..22; stepped roots endz27;6mm leaves occupy x5.5..11.5,z27..49; lower ledges reach x15.5,z52..54. The2.4mm rear gaps end at the real rigid backstop x3.1. Axial shoulders reach x13.5 over y=±22..26,z27..49.

The20mm centre pad has a root bridge x3.1..5.5,z19..27,6mm leaf x5.5..11.5,z19..51, and a forward distal ramp reaching x15.5 atz54. Three keys have7.4mm heads,4mm depth x−0.9..3.1, outer5.6mm necks,z18..26 and centre4.4mm neck,z19..27. The backing leaves1.6mm behind key heads. Convex edges round inward0.3mm.

Every entry lip is at or behind x15.5, giving the full80mm nominal aperture at maximum Hand opening. Main leaf faces admit88mm and axial shoulders84mm, but the entry limits usable object width. Empty pad contact begins near stock width0; real homing/contact calibration remains mandatory. The central ramp captures the small-object cross-section about6mm below the centre, so its position-width rule includes a sphere cross-section correction with a nonnegative width bound. This is still sensitive to shape and size uncertainty.

## Hardware and mounting

Per pair: four original-specification **M4x12 screws at2.25Nm**, four M4 ISO7093-1 steel washers (nominal12mm OD,1mm thick), and four **SPIROL CL200 M4 steel compression limiters**, cut and precision-lapped to each local web. Hardware is purchased separately and is not plastic geometry.

Before printing/installing, measure the real post (model x0..6,y+-10.5,z+-7.5),12mm screw-axis spacing, seating face, blind-thread depth, housing and driver clearances. The CAD lower-post clearance begins at z7.7; verify it. Finish clamp web flat/parallel to6.00+/-0.02mm and limiter bores6.50–6.60mm. Installed limiter ID must be at least4.5mm.

For each measured local web thickness H, finish its matched limiter to H−0.020mm and verify H−L in0.010..0.030mm including measurement uncertainty. A stock-length limiter is unsuitable. Use a supported fixture; washers must bear on both limiter and polymer. The screw projects about5mm beyond a6mm web plus1mm washer: measure useful engagement and bottom clearance on the actual carriage. Do not infer blind thread depth from the6mm post model. The head may project from the rear recess; verify tool access. With the robot de-energized, seat flat and torque using a calibrated tool. Inspect seating, cracks and preload retention. If precision finishing or hardware is unavailable, this mount is not ready to install.

## Qualification and control

Use a carriage-replica fixture and sacrificial prints to measure pad force/displacement, hysteresis, bottoming, key pull-out, shear/peel, off-axis torque, sustained creep and cycling. Establish actual working loads with a documented safety margin; do not proof-load the robot beyond its limits. Bambu TPU90 typical tensile modulus5.3MPa XY/4.4MPa Z differs from the lab class15MPa. The fixed collision model does not resolve the leaves bending into their stops. PETG HF typical XY/Z strengths34/23MPa are supplier specimen results, not a certified allowable for this unannealed co-print. No previous version's beam estimate is a qualification of this body.

`grasp.py` is a deterministic rule over the supplied measured task sizes. It never reads trial IDs or seeds. It requests an end-based press pickup and85% measured stroke; wide/infeasible squeeze configurations are left untouched and count as failures. Accepted attempts can still miss, slip, crush objects or hit the rest/counter. Its nominal geometric margins do not guarantee clearance. Additional perturbation and summed-wrench reports are research evidence, not controller certification.

The real adapter must rehome after the finger change, measure width-to-gap conversion, transform the object-frame poses and check full-arm reach/collision paths. The lab close-to-contact default must not be copied as a real `grasp(width=0)` call: libfranka needs calibrated expected width, suitable epsilon tolerances, speed, success/is_grasped checks and stop behavior. Implement sensed contact/slip/load aborts. No hardware adapter or controller execution record exists here.

Respect the Hand30–70N command and80mm travel envelope. The lab3.78Nm carriage limit is70N×54mm, not a separately quoted manufacturer torque rating. Sum simultaneous contact moments and support forces, with gravity/inertia margin; a largest-single-contact metric is insufficient. Use modeled2mm deflection/material stress limits only as screening. The independent audit found additional failed attempts that the lab labeled successful.

Initial tests: guarded pinch/drop zones, drop containment, low arm speed, operator stop; dry20–35C fixtures, no hot cookware, hob, steam, splashes or washdown.35C is a conservative prototype restriction, not a TPU heat rating. Hand isIP20. Weigh both fingers, hardware, Hand and adapters, transform both CAD mass/CoG/inertias to the flange frame, and configure the complete end effector.0.730kg Hand+0.120kg printed pair+0.020kg hardware+2kg object leaves only0.130kg for other equipment under the FR3 arm3kg rating; long-handle moments need separate checking.

Food-for-consumption release is **unmet**. Grade/color declarations, finished-article migration, cleanability, traceability and GMP records are absent. Open gaps and FDM layers retain contamination. Use sacrificial discarded food only during qualification, and replace the entire co-print after tearing/delamination/contamination. No invisible coating or generic food-safe PETG/TPU claim is made.

## Sources

- Franka Hand R50010/1.2, sections5.1,5.2.4,7.1,7.3: https://franka.de/hubfs/Product%20Manual%20Franka%20Hand_R50010_1.2_EN.pdf
- Bambu H2C manual6.2,8.4: https://csm.bblcdn.com/hub/eff78da43720461787dc8bbe5fa0372d.pdf
- Current Bambu TPU90/PETG compatibility and printing guidance: https://bambu-lab-jp.myshopify.com/en/products/tpu-85a-tpu-90a
- Bambu TPU90 TDS V1.0: https://cdn.shopify.com/s/files/1/0645/5876/0155/files/Bambu_TPU_90A_Technical_Data_Sheet_582bf8f6-1f0a-474c-aeda-9e72af3689dc.pdf?v=1741314674
- PETG HF TDS: https://store.bblcdn.com/3a230e260a3a47c2b0db0156e07eef91.pdf
- SPIROL compression limiter design: https://www.spirol.com/assets/files/cmpl-compression-limiter-design-guide-us.pdf
- Further regulation, printer and controller sources with applicability are in requirements.yaml.


## Evidence limits and cycle behavior

A corrected dimensional squeeze release requests80% of the estimated natural spread at the selected lever, separately from the transport preload. The observer checks actual tip gaps after both cycles: commanding a release is insufficient if the tool stays closed. Cracker arms that fit no observed slot pair may be refused; noisy observations can misclassify placement. The press rule uses85% of measured stroke following the lab's full-length clearance lift. It does not use the obsolete precompression correction.

The soft-object width is an empirical bounded preload schedule scaled with weight, NOT a physically calibrated spring model. Its inherited1.5N/mm normalization must not be mistaken for the stiffness of these pads. Material-class contact softness does not model a leaf bending, its changing contact position, or snap-in-key deformation. Actual force/displacement, hysteresis and surface friction must be measured before these widths are transferred.

The public simulation position servo can deliver less than the requested force cap at small jaw widths. Moving the contact faces outward changes both aperture and attainable simulated normal force at the same object diameter. No illegal negative width or material adjustment is used to compensate. Real force-controlled behavior needs its own adapter and measured force verification.

See AUDIT.md for candidate-specific evidence. Lab success is a screening result; dangerous support contact, excessive summed carriage wrench, failed reopening or object damage remains a failed real application even if a narrower lab predicate passes. Full-arm reach, scene calibration and unexpected-contact stops have not been validated.


## Conservative slot-seating guard

This revision refuses squeeze pickup when observed arm width plus two length-observation-error margins (each3%+1.5mm) exceeds slot width minus1mm usable clearance. An uncertain counter/ridge-top placement is not assumed safe. This dimensional refusal can cost tool successes and does not solve all support collisions. Geometry, other task policies and actual release behavior are unchanged. Its direct predecessor was validated; this guarded policy must be reported separately because validation slots are exhausted.


Press refusal additionally checks estimated post-stroke clearance for the full Hand housing above the rest, with5mm geometric clearance and two observation-error margins. It retains the 85% press-depth rule; the current accepted pickup uses the smaller opening and 2 mm height adjustment described below. This reduces coverage, and does not guarantee against pickup collisions or slip-induced rotation.


## Inspection drawing and evidence

sections.svg is generated from actual printable STL cross-sections. AUDIT.md and audit_evidence contain current guarded-policy results and limitations. The final physical release remains blocked by unsafe trajectories, missing controller/calibration evidence, untested manufacturing/retention and food-contact qualification.


## Press pickup clearance revision

Accepted press pickup uses tool-frame TCP z=0 (2 mm above the earlier rule) and a pre-opening equal to measured handle diameter + two measurement margins (3% diameter + 1.5 mm each) + 10 mm clearance, capped at 80 mm. Force, stroke fraction, end offset and dimensional refusals are unchanged. All non-press rules and all print geometry are unchanged. This reduces observed counter contacts but does not guarantee clearance or provide sensed contact stops. See AUDIT.md for the new 64-press and extra-noise evidence. This revision intentionally changes accepted press approaches; all other task commands are unchanged.

## Shop setup (2026-11-05): 0.6 mm nozzles, TPU 85A

Printed on the shop's H2C with 0.6 mm nozzles and **Bambu TPU 85A** for the pads (Bambu's TPU 85A profile allows only
0.6 and 0.8 mm nozzles). Changes from the profile above: PETG HF 0.6 mm profile, 0.20-0.24 mm layers (not the 0.6
nozzle's default 0.30: the keys, the 1.6 mm backing behind them and the 2.4 mm rear gaps need the finer layers),
0.62 mm line width, 4 walls (about 2.5 mm, as six 0.42 mm walls were), 100% infill; TPU 85A at 225 C with its profile's
2.2 mm3/s volumetric limit (slower than 90A's 2.8), external/top feed only (85A cannot go through an AMS), dried 70 C
8 h. The 0.3 mm edge rounds are lost at this line width (cosmetic). 85A is softer and grippier than the 90A the lab
designed with (modulus about 10 vs 15 MPa in the lab's classes): the pads' leaves deflect more under load and the keys
hold less firmly; check key pull-out on the test coupon. The pick lab's simulated pads use 85A (materials.yaml).

## Revision 2 (2026-11-06): pads flush on the PETG, printed flat

The first co-print failed. In revision 2 the 2.4 mm rear gaps behind the pad leaves are filled with TPU: each leaf's
rear outline extruded back to the PETG backing face (x 3.1) and merged into the pad (`parts/pad_filled.stl`, 14.0 cm3
per finger, was 11.5; the TPU rests on the PETG over 1268 mm2, was 223; none of it outside the body's outline, none
of it inside the body). The body is unchanged. 85A is soft enough that the leaves don't need to flex into a gap: the
finger is stiffer than the lab tested, its give now the TPU's own.

Printed flat on its back: `parts/finger_body_flat.stl` and `parts/finger_pads_flat.stl` (the same parts turned so the
finger's -x is on the bed, its length along X); the project is `franka_fingers_flat_H2C_bambu.3mf` (kept outside the
repo). PETG first, the TPU from 9.1 mm up on top of it. Only the mount block and the back rib touch the bed (451
mm2): tree supports under the backing plate (about 2000 mm2, 8 mm up) and a 5 mm brim. Beam interlocking on (1.2 mm
beams, 2 layers, depth 2, 2 from the edges) weaves the TPU into the PETG across the whole contact face; the three root
keys are unchanged. The simulation's collision pieces are unchanged (the gripping surfaces are the same).
