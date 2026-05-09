#include "Viewer.hpp"

#include <chrono>
#include <cmath>
#include <thread>

#include <pangolin/pangolin.h>

static bool isFinitePose(const cv::Matx44d& T) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(T(r, c))) return false;
        }
    }
    return true;
}

Viewer::Viewer(const Calibration& calib, std::size_t max_points)
    : calib_(calib), max_points_(max_points) {}

Viewer::~Viewer() {
    requestStop();
    join();
}

void Viewer::start() {
    stop_requested_ = false;
    thread_ = std::thread(&Viewer::renderLoop, this);
}

void Viewer::requestStop() {
    stop_requested_ = true;
}

void Viewer::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Viewer::pushUpdate(
    const cv::Matx44d& T_world_cam,
    const std::vector<cv::Vec3f>& world_points,
    const cv::Mat& left_image
) {
    std::lock_guard<std::mutex> lock(mutex_);

    trajectory_.push_back(T_world_cam);

    for (const auto& p : world_points) {
        map_points_.push_back(p);
    }
    while (map_points_.size() > max_points_) {
        map_points_.pop_front();
    }

    if (!left_image.empty()) {
        left_image.copyTo(current_image_);
        image_width_ = left_image.cols;
        image_height_ = left_image.rows;
    }
}

static void drawFrustum(
    const cv::Matx44d& T,
    float size,
    float r, float g, float b,
    float line_width
) {
    const float w = 0.5f * size;
    const float h = 0.3f * size;
    const float z = size;

    auto tx = [&](double cx, double cy, double cz) {
        cv::Vec4d ph(cx, cy, cz, 1.0);
        cv::Vec4d pw = T * ph;
        return cv::Vec3f((float)pw[0], (float)pw[1], (float)pw[2]);
    };

    cv::Vec3f apex = tx(0, 0, 0);
    cv::Vec3f tl = tx(-w, -h, z);
    cv::Vec3f tr = tx( w, -h, z);
    cv::Vec3f br = tx( w,  h, z);
    cv::Vec3f bl = tx(-w,  h, z);

    glLineWidth(line_width);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex3f(apex[0], apex[1], apex[2]); glVertex3f(tl[0], tl[1], tl[2]);
    glVertex3f(apex[0], apex[1], apex[2]); glVertex3f(tr[0], tr[1], tr[2]);
    glVertex3f(apex[0], apex[1], apex[2]); glVertex3f(br[0], br[1], br[2]);
    glVertex3f(apex[0], apex[1], apex[2]); glVertex3f(bl[0], bl[1], bl[2]);
    glVertex3f(tl[0], tl[1], tl[2]); glVertex3f(tr[0], tr[1], tr[2]);
    glVertex3f(tr[0], tr[1], tr[2]); glVertex3f(br[0], br[1], br[2]);
    glVertex3f(br[0], br[1], br[2]); glVertex3f(bl[0], bl[1], bl[2]);
    glVertex3f(bl[0], bl[1], bl[2]); glVertex3f(tl[0], tl[1], tl[2]);
    glEnd();
}

static void drawGrid(float size, int divisions) {
    const float step = size / divisions;
    glLineWidth(1.0f);
    glColor3f(0.22f, 0.22f, 0.25f);
    glBegin(GL_LINES);
    for (int i = -divisions; i <= divisions; ++i) {
        float p = i * step;
        glVertex3f(p, 0.0f, -size); glVertex3f(p, 0.0f, size);
        glVertex3f(-size, 0.0f, p); glVertex3f(size, 0.0f, p);
    }
    glEnd();
}

void Viewer::renderLoop() {
    pangolin::CreateWindowAndBind("Stereo VO Viewer", 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -30, -30, 0, 0, 30, pangolin::AxisNegY)
    );

    const int ui_width = 175;

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(ui_width));

    pangolin::Var<bool>  show_points("ui.show_points", true, true);
    pangolin::Var<bool>  show_trajectory("ui.show_trajectory", true, true);
    pangolin::Var<bool>  show_frustums("ui.show_frustums", true, true);
    pangolin::Var<int>   frustum_stride("ui.frustum_stride", 1, 1, 20);
    pangolin::Var<bool>  show_grid("ui.show_grid", true, true);
    pangolin::Var<bool>  follow_camera("ui.follow_camera", false, true);
    pangolin::Var<float> point_size("ui.point_size", 2.0f, 1.0f, 8.0f);

    const float img_strip_height = 0.20f;

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(img_strip_height, 1.0, pangolin::Attach::Pix(ui_width), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0.0, img_strip_height, pangolin::Attach::Pix(ui_width), 1.0);

    pangolin::GlTexture image_texture;
    bool texture_initialized = false;

    while (!stop_requested_ && !pangolin::ShouldQuit()) {
        std::vector<cv::Matx44d> traj_copy;
        std::vector<cv::Vec3f> points_copy;
        cv::Mat image_copy;
        int img_w = image_width_;
        int img_h = image_height_;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            traj_copy = trajectory_;
            points_copy.assign(map_points_.begin(), map_points_.end());
            current_image_.copyTo(image_copy);
            img_w = image_width_;
            img_h = image_height_;
        }

        if (follow_camera && !traj_copy.empty() && isFinitePose(traj_copy.back())) {
            const auto& T = traj_copy.back();
            double cx = T(0, 3), cy = T(1, 3), cz = T(2, 3);
            double fx_ = T(0, 2), fy_ = T(1, 2), fz_ = T(2, 2);
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                cx - 6.0 * fx_, cy - 6.0 * fy_ - 2.0, cz - 6.0 * fz_,
                cx + 10.0 * fx_, cy + 10.0 * fy_, cz + 10.0 * fz_,
                pangolin::AxisNegY
            ));
        }

        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        if (show_grid) {
            drawGrid(50.0f, 50);
        }

        if (show_trajectory && traj_copy.size() >= 2) {
            glColor3f(1.0f, 1.0f, 0.0f);
            glLineWidth(2.0f);
            glBegin(GL_LINE_STRIP);
            for (const auto& T : traj_copy) {
                if (!isFinitePose(T)) continue;
                glVertex3f((float)T(0, 3), (float)T(1, 3), (float)T(2, 3));
            }
            glEnd();
        }

        if (show_frustums && !traj_copy.empty()) {
            int stride = std::max(1, (int)frustum_stride);
            for (size_t i = 0; i + 1 < traj_copy.size(); i += stride) {
                if (!isFinitePose(traj_copy[i])) continue;
                float t = traj_copy.size() > 1
                    ? (float)i / (float)(traj_copy.size() - 1)
                    : 0.0f;
                drawFrustum(
                    traj_copy[i],
                    0.4f,
                    0.30f + 0.50f * t,
                    0.45f,
                    1.00f - 0.50f * t,
                    1.0f
                );
            }
        }

        if (!traj_copy.empty() && isFinitePose(traj_copy.back())) {
            drawFrustum(traj_copy.back(), 1.2f, 1.0f, 0.25f, 0.25f, 2.5f);
        }

        if (show_points && !points_copy.empty()) {
            glPointSize((float)point_size);
            glBegin(GL_POINTS);
            for (size_t i = 0; i < points_copy.size(); ++i) {
                float t = (float)i / (float)points_copy.size();
                glColor3f(0.4f + 0.6f * t, 0.7f, 1.0f - 0.4f * t);
                glVertex3f(points_copy[i][0], points_copy[i][1], points_copy[i][2]);
            }
            glEnd();
        }

        if (!image_copy.empty()) {
            if (!texture_initialized
                || (int)image_texture.width != img_w
                || (int)image_texture.height != img_h) {
                image_texture.Reinitialise(
                    img_w, img_h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE
                );
                texture_initialized = true;
            }

            cv::Mat rgb;
            if (image_copy.channels() == 1) {
                cv::cvtColor(image_copy, rgb, cv::COLOR_GRAY2RGB);
            } else {
                cv::cvtColor(image_copy, rgb, cv::COLOR_BGR2RGB);
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            image_texture.Upload(rgb.data, GL_RGB, GL_UNSIGNED_BYTE);

            d_image.Activate();
            glColor3f(1.0f, 1.0f, 1.0f);
            image_texture.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
