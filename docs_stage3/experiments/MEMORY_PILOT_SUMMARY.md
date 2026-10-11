# Memory-policy pilot: actual 15-case matrix

All 15 cases completed without program errors. Each configuration has one seed (7); this is descriptive evidence, not a statistically significant success-rate estimate. Different FOV/observation periods are kept separate. Full registered configs, failures and traces remain in the parent result directory.

| Sensor | Memory | Navigation | Actual minimum gap (m) | Unsafe forecast | Lost IDs | Conditional deletions | Max extra uncertainty (m) | P95 (ms) | Deadline misses |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fov120x30_obs1 | hard4 | success | 0.508115 | 3 | 3 | 0 | 0.000 | 1547.865 | 0 |
| fov120x30_obs1 | hard8 | success | 0.456696 | 8 | 3 | 0 | 0.000 | 1645.867 | 0 |
| fov120x30_obs1 | hard12 | success | 0.456696 | 6 | 3 | 0 | 0.000 | 1786.705 | 0 |
| fov120x30_obs1 | retain_uncertain | timeout | 0.456696 | 60 | 2 | 0 | 295.000 | 1761.705 | 0 |
| fov120x30_obs1 | conditional | timeout | 0.456696 | 60 | 2 | 0 | 295.000 | 1651.634 | 0 |
| fov90x6_obs1 | hard4 | obstacle_collision | -0.017002 | 2 | 3 | 0 | 0.000 | 1568.979 | 0 |
| fov90x6_obs1 | hard8 | obstacle_collision | -0.016946 | 3 | 3 | 0 | 0.000 | 1592.279 | 0 |
| fov90x6_obs1 | hard12 | obstacle_collision | -0.016946 | 3 | 3 | 0 | 0.000 | 1712.608 | 0 |
| fov90x6_obs1 | retain_uncertain | obstacle_collision | -0.016946 | 3 | 3 | 0 | 10.000 | 1814.962 | 1 |
| fov90x6_obs1 | conditional | obstacle_collision | -0.016946 | 3 | 3 | 0 | 10.000 | 1763.261 | 0 |
| fov120x30_obs2 | hard4 | success | 0.508115 | 3 | 3 | 0 | 0.000 | 1820.928 | 0 |
| fov120x30_obs2 | hard8 | success | 0.456696 | 8 | 3 | 0 | 0.000 | 1888.628 | 0 |
| fov120x30_obs2 | hard12 | success | 0.456696 | 7 | 3 | 0 | 0.000 | 1987.054 | 3 |
| fov120x30_obs2 | retain_uncertain | timeout | 0.456696 | 60 | 2 | 0 | 295.000 | 4298.148 | 44 |
| fov120x30_obs2 | conditional | timeout | 0.456696 | 60 | 2 | 0 | 295.000 | 4211.989 | 65 |

The original hard4 wide-FOV configuration reproduces the earlier minimum gap 0.508114839 m. Merely extending hard expiry to 8 or 12 s does not remove that violation. With narrow 90 x 6 degree FOV, all five policies collide in this fixed speed-scale1.5 scene. Retaining stale targets with increasing uncertainty produces four wide-FOV timeouts and large conservative envelopes. Conditional deletion count is shown explicitly; its declared reachability assumptions are not a calibrated guarantee.

The original 1 m criterion, collision geometry and goal tolerance remain unchanged. All 15 cases fail finite-horizon safety acceptance; arrivals are still recorded independently. Timing was measured with concurrent development/experiment processes and synchronous offline execution, so it is not dedicated-machine real-time performance. Obstacle prediction RMSE is marked pending independent same-anchor offline assessment in the original matrix CSV.
