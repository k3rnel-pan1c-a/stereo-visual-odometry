#pragma once

struct Config {
        int max_frames = 1500;

        int orb_features = 3000;

        double stereo_max_y_diff = 1.5;
        double min_disparity = 2.0;
        double min_depth = 1.0;
        double max_depth = 80.0;

        int min_pnp_points = 30;
        int min_pnp_inliers = 40;
        double pnp_reproj_error = 3.0;
        int pnp_iterations = 100;
        double pnp_confidence = 0.99;

        double max_match_distance = 50.0;
};
