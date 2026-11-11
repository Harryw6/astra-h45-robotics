#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace fs = std::filesystem;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

namespace {

std::atomic<bool> running{true};

void handle_signal(int) { running.store(false); }

int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <typename Range>
void append_csv(std::ostream& out, const Range& values) {
  for (const auto& value : values) out << ',' << +value;
}

template <typename Range>
void write_json_array(std::ostream& out, const Range& values) {
  out << '[';
  bool first = true;
  for (const auto& value : values) {
    if (!first) out << ',';
    first = false;
    out << +value;
  }
  out << ']';
}

struct WirelessSnapshot {
  bool valid = false;
  int64_t monotonic = 0;
  int64_t wall = 0;
  uint64_t seq = 0;
  float lx = 0;
  float ly = 0;
  float rx = 0;
  float ry = 0;
  uint16_t keys = 0;
};

struct SportSnapshot {
  bool valid = false;
  int64_t monotonic = 0;
  int64_t wall = 0;
  uint64_t seq = 0;
  uint32_t robot_sec = 0;
  uint32_t robot_nanosec = 0;
  uint32_t error_code = 0;
  uint8_t mode = 0;
  uint8_t gait_type = 0;
  float progress = 0;
  float foot_raise_height = 0;
  std::array<float, 3> position{};
  float body_height = 0;
  std::array<float, 3> velocity{};
  float yaw_speed = 0;
  std::array<float, 4> quaternion{};
  std::array<float, 3> gyroscope{};
  std::array<float, 3> accelerometer{};
  std::array<float, 3> rpy{};
  std::array<float, 4> range_obstacle{};
  std::array<float, 4> foot_force{};
};

struct LowSnapshot {
  bool valid = false;
  int64_t monotonic = 0;
  int64_t wall = 0;
  uint64_t seq = 0;
  uint32_t tick = 0;
  float power_v = 0;
  float power_a = 0;
  std::array<float, 4> quaternion{};
  std::array<float, 3> gyroscope{};
  std::array<float, 3> accelerometer{};
  std::array<float, 3> rpy{};
  std::array<float, 4> foot_force{};
  std::array<float, 4> foot_force_est{};
  std::array<float, 20> motor_q{};
  std::array<float, 20> motor_dq{};
  std::array<float, 20> motor_tau{};
};

template <typename Target, typename Source>
void copy_values(Target& target, const Source& source) {
  const size_t count = std::min(target.size(), source.size());
  for (size_t i = 0; i < count; ++i) target[i] = source[i];
}

class Bridge {
 public:
  explicit Bridge(const fs::path& raw_dir)
      : raw_dir_(raw_dir),
        wireless_file_(raw_dir / "wireless_controller.csv"),
        sport_file_(raw_dir / "sport_mode_state.csv"),
        low_file_(raw_dir / "low_state.csv") {
    if (!wireless_file_ || !sport_file_ || !low_file_) {
      throw std::runtime_error("failed to open Go2 output files");
    }
    wireless_file_ << std::setprecision(17);
    sport_file_ << std::setprecision(17);
    low_file_ << std::setprecision(17);
    write_headers();
  }

  void start() {
    wireless_sub_.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
            "rt/wirelesscontroller"));
    sport_sub_.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
            "rt/sportmodestate"));
    low_sub_.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
        "rt/lowstate"));
    wireless_sub_->InitChannel(
        std::bind(&Bridge::on_wireless, this, std::placeholders::_1), 32);
    sport_sub_->InitChannel(
        std::bind(&Bridge::on_sport, this, std::placeholders::_1), 32);
    low_sub_->InitChannel(
        std::bind(&Bridge::on_low, this, std::placeholders::_1), 32);
  }

  void flush() {
    std::scoped_lock lock(wireless_mutex_, sport_mutex_, low_mutex_);
    wireless_file_.flush();
    sport_file_.flush();
    low_file_.flush();
  }

  void write_snapshot() {
    WirelessSnapshot wireless;
    SportSnapshot sport;
    LowSnapshot low;
    {
      std::scoped_lock lock(wireless_mutex_, sport_mutex_, low_mutex_);
      wireless = wireless_snapshot_;
      sport = sport_snapshot_;
      low = low_snapshot_;
    }
    const fs::path tmp = raw_dir_ / ".go2_snapshot.json.tmp";
    const fs::path dst = raw_dir_ / "go2_snapshot.json";
    std::ofstream out(tmp);
    if (!out) return;
    out << std::setprecision(17) << '{';
    write_wireless_json(out, wireless);
    out << ',';
    write_sport_json(out, sport);
    out << ',';
    write_low_json(out, low);
    out << "}\n";
    out.close();
    std::error_code ec;
    fs::rename(tmp, dst, ec);
  }

  uint64_t wireless_count() const { return wireless_count_.load(); }
  uint64_t sport_count() const { return sport_count_.load(); }
  uint64_t low_count() const { return low_count_.load(); }

 private:
  void write_headers() {
    wireless_file_ << "monotonic_ns,wall_time_ns,seq,lx,ly,rx,ry,keys\n";
    sport_file_
        << "monotonic_ns,wall_time_ns,seq,robot_sec,robot_nanosec,error_code,"
           "mode,gait_type,progress,foot_raise_height,"
           "position_x,position_y,position_z,body_height,"
           "velocity_x,velocity_y,velocity_z,yaw_speed,"
           "quat_w,quat_x,quat_y,quat_z,"
           "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,"
           "roll,pitch,yaw,range_0,range_1,range_2,range_3,"
           "foot_force_0,foot_force_1,foot_force_2,foot_force_3\n";
    low_file_
        << "monotonic_ns,wall_time_ns,seq,tick,power_v,power_a,"
           "quat_w,quat_x,quat_y,quat_z,"
           "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,"
           "roll,pitch,yaw,foot_force_0,foot_force_1,foot_force_2,"
           "foot_force_3,foot_force_est_0,foot_force_est_1,"
           "foot_force_est_2,foot_force_est_3";
    for (int i = 0; i < 40; ++i) low_file_ << ",wireless_raw_" << i;
    for (int i = 0; i < 20; ++i) {
      low_file_ << ",motor_" << i << "_mode"
                << ",motor_" << i << "_q"
                << ",motor_" << i << "_dq"
                << ",motor_" << i << "_ddq"
                << ",motor_" << i << "_tau_est"
                << ",motor_" << i << "_temperature"
                << ",motor_" << i << "_lost";
    }
    low_file_ << '\n';
  }

  void on_wireless(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::WirelessController_*>(
            message);
    WirelessSnapshot value;
    value.valid = true;
    value.monotonic = monotonic_ns();
    value.wall = wall_ns();
    value.seq = wireless_count_.fetch_add(1);
    value.lx = msg.lx();
    value.ly = msg.ly();
    value.rx = msg.rx();
    value.ry = msg.ry();
    value.keys = msg.keys();
    std::lock_guard<std::mutex> lock(wireless_mutex_);
    wireless_snapshot_ = value;
    wireless_file_ << value.monotonic << ',' << value.wall << ',' << value.seq
                   << ',' << value.lx << ',' << value.ly << ',' << value.rx
                   << ',' << value.ry << ',' << value.keys << '\n';
  }

  void on_sport(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
    const auto& imu = msg.imu_state();
    SportSnapshot value;
    value.valid = true;
    value.monotonic = monotonic_ns();
    value.wall = wall_ns();
    value.seq = sport_count_.fetch_add(1);
    value.robot_sec = msg.stamp().sec();
    value.robot_nanosec = msg.stamp().nanosec();
    value.error_code = msg.error_code();
    value.mode = msg.mode();
    value.gait_type = msg.gait_type();
    value.progress = msg.progress();
    value.foot_raise_height = msg.foot_raise_height();
    copy_values(value.position, msg.position());
    value.body_height = msg.body_height();
    copy_values(value.velocity, msg.velocity());
    value.yaw_speed = msg.yaw_speed();
    copy_values(value.quaternion, imu.quaternion());
    copy_values(value.gyroscope, imu.gyroscope());
    copy_values(value.accelerometer, imu.accelerometer());
    copy_values(value.rpy, imu.rpy());
    copy_values(value.range_obstacle, msg.range_obstacle());
    copy_values(value.foot_force, msg.foot_force());
    std::lock_guard<std::mutex> lock(sport_mutex_);
    sport_snapshot_ = value;
    sport_file_ << value.monotonic << ',' << value.wall << ',' << value.seq
                << ',' << value.robot_sec << ',' << value.robot_nanosec << ','
                << value.error_code << ',' << +value.mode << ','
                << +value.gait_type << ',' << value.progress << ','
                << value.foot_raise_height;
    append_csv(sport_file_, value.position);
    sport_file_ << ',' << value.body_height;
    append_csv(sport_file_, value.velocity);
    sport_file_ << ',' << value.yaw_speed;
    append_csv(sport_file_, value.quaternion);
    append_csv(sport_file_, value.gyroscope);
    append_csv(sport_file_, value.accelerometer);
    append_csv(sport_file_, value.rpy);
    append_csv(sport_file_, value.range_obstacle);
    append_csv(sport_file_, value.foot_force);
    sport_file_ << '\n';
  }

  void on_low(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::LowState_*>(message);
    const auto& imu = msg.imu_state();
    LowSnapshot value;
    value.valid = true;
    value.monotonic = monotonic_ns();
    value.wall = wall_ns();
    value.seq = low_count_.fetch_add(1);
    value.tick = msg.tick();
    value.power_v = msg.power_v();
    value.power_a = msg.power_a();
    copy_values(value.quaternion, imu.quaternion());
    copy_values(value.gyroscope, imu.gyroscope());
    copy_values(value.accelerometer, imu.accelerometer());
    copy_values(value.rpy, imu.rpy());
    copy_values(value.foot_force, msg.foot_force());
    copy_values(value.foot_force_est, msg.foot_force_est());
    for (size_t i = 0; i < value.motor_q.size() && i < msg.motor_state().size();
         ++i) {
      value.motor_q[i] = msg.motor_state()[i].q();
      value.motor_dq[i] = msg.motor_state()[i].dq();
      value.motor_tau[i] = msg.motor_state()[i].tau_est();
    }
    std::lock_guard<std::mutex> lock(low_mutex_);
    low_snapshot_ = value;
    low_file_ << value.monotonic << ',' << value.wall << ',' << value.seq
              << ',' << value.tick << ',' << value.power_v << ','
              << value.power_a;
    append_csv(low_file_, imu.quaternion());
    append_csv(low_file_, imu.gyroscope());
    append_csv(low_file_, imu.accelerometer());
    append_csv(low_file_, imu.rpy());
    append_csv(low_file_, msg.foot_force());
    append_csv(low_file_, msg.foot_force_est());
    append_csv(low_file_, msg.wireless_remote());
    for (const auto& motor : msg.motor_state()) {
      low_file_ << ',' << +motor.mode() << ',' << motor.q() << ',' << motor.dq()
                << ',' << motor.ddq() << ',' << motor.tau_est() << ','
                << +motor.temperature() << ',' << motor.lost();
    }
    low_file_ << '\n';
  }

  static void write_wireless_json(std::ostream& out,
                                  const WirelessSnapshot& value) {
    out << "\"wireless_controller\":{\"valid\":"
        << (value.valid ? "true" : "false")
        << ",\"monotonic_ns\":" << value.monotonic
        << ",\"wall_time_ns\":" << value.wall << ",\"seq\":" << value.seq
        << ",\"lx\":" << value.lx << ",\"ly\":" << value.ly
        << ",\"rx\":" << value.rx << ",\"ry\":" << value.ry
        << ",\"keys\":" << value.keys << '}';
  }

  static void write_sport_json(std::ostream& out,
                               const SportSnapshot& value) {
    out << "\"sport_mode_state\":{\"valid\":"
        << (value.valid ? "true" : "false")
        << ",\"monotonic_ns\":" << value.monotonic
        << ",\"wall_time_ns\":" << value.wall << ",\"seq\":" << value.seq
        << ",\"robot_sec\":" << value.robot_sec
        << ",\"robot_nanosec\":" << value.robot_nanosec
        << ",\"error_code\":" << value.error_code
        << ",\"mode\":" << +value.mode << ",\"gait_type\":"
        << +value.gait_type << ",\"progress\":" << value.progress
        << ",\"foot_raise_height\":" << value.foot_raise_height
        << ",\"position\":";
    write_json_array(out, value.position);
    out << ",\"body_height\":" << value.body_height << ",\"velocity\":";
    write_json_array(out, value.velocity);
    out << ",\"yaw_speed\":" << value.yaw_speed << ",\"quaternion\":";
    write_json_array(out, value.quaternion);
    out << ",\"gyroscope\":";
    write_json_array(out, value.gyroscope);
    out << ",\"accelerometer\":";
    write_json_array(out, value.accelerometer);
    out << ",\"rpy\":";
    write_json_array(out, value.rpy);
    out << ",\"range_obstacle\":";
    write_json_array(out, value.range_obstacle);
    out << ",\"foot_force\":";
    write_json_array(out, value.foot_force);
    out << '}';
  }

  static void write_low_json(std::ostream& out, const LowSnapshot& value) {
    out << "\"low_state\":{\"valid\":"
        << (value.valid ? "true" : "false")
        << ",\"monotonic_ns\":" << value.monotonic
        << ",\"wall_time_ns\":" << value.wall << ",\"seq\":" << value.seq
        << ",\"tick\":" << value.tick << ",\"power_v\":" << value.power_v
        << ",\"power_a\":" << value.power_a << ",\"quaternion\":";
    write_json_array(out, value.quaternion);
    out << ",\"gyroscope\":";
    write_json_array(out, value.gyroscope);
    out << ",\"accelerometer\":";
    write_json_array(out, value.accelerometer);
    out << ",\"rpy\":";
    write_json_array(out, value.rpy);
    out << ",\"foot_force\":";
    write_json_array(out, value.foot_force);
    out << ",\"foot_force_est\":";
    write_json_array(out, value.foot_force_est);
    out << ",\"motor_q\":";
    write_json_array(out, value.motor_q);
    out << ",\"motor_dq\":";
    write_json_array(out, value.motor_dq);
    out << ",\"motor_tau_est\":";
    write_json_array(out, value.motor_tau);
    out << '}';
  }

  fs::path raw_dir_;
  std::ofstream wireless_file_;
  std::ofstream sport_file_;
  std::ofstream low_file_;
  std::mutex wireless_mutex_;
  std::mutex sport_mutex_;
  std::mutex low_mutex_;
  WirelessSnapshot wireless_snapshot_;
  SportSnapshot sport_snapshot_;
  LowSnapshot low_snapshot_;
  std::atomic<uint64_t> wireless_count_{0};
  std::atomic<uint64_t> sport_count_{0};
  std::atomic<uint64_t> low_count_{0};
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_>
      wireless_sub_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sport_sub_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> low_sub_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " NETWORK_INTERFACE RAW_OUTPUT_DIRECTORY\n";
    return 2;
  }
  const std::string network_interface = argv[1];
  const fs::path raw_dir = argv[2];
  fs::create_directories(raw_dir);
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  try {
    ChannelFactory::Instance()->Init(0, network_interface);
    Bridge bridge(raw_dir);
    bridge.start();
    std::cout << "Go2 read-only bridge recording on " << network_interface
              << " into " << raw_dir << std::endl;
    auto next_flush = std::chrono::steady_clock::now();
    while (running.load()) {
      bridge.write_snapshot();
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_flush) {
        bridge.flush();
        next_flush = now + std::chrono::seconds(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    bridge.write_snapshot();
    bridge.flush();
    std::cout << "Go2 bridge stopped: wireless=" << bridge.wireless_count()
              << " sport=" << bridge.sport_count()
              << " low=" << bridge.low_count() << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "go2_state_bridge: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
