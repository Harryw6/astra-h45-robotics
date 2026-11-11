#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

namespace {

std::atomic<bool> running{true};
std::string socket_path;

void handle_signal(int) { running.store(false); }

int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string json_float(float value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(9) << value;
  return out.str();
}

void log_motion(const char* event, const char* source, uint64_t sequence,
                int rc, float vx, float vy, float vyaw, int64_t started_ns) {
  std::cout << std::setprecision(9) << "{\"event\":\"" << event
            << "\",\"source\":\"" << source << "\",\"sequence\":" << sequence
            << ",\"call_started_ns\":" << started_ns
            << ",\"monotonic_ns\":" << monotonic_ns()
            << ",\"sdk_rc\":" << rc << ",\"velocity\":["
            << json_float(vx) << ',' << json_float(vy) << ','
            << json_float(vyaw) << "]}" << std::endl;
}

void remove_socket() {
  if (!socket_path.empty()) ::unlink(socket_path.c_str());
}

struct State {
  bool valid = false;
  int64_t monotonic_ns = 0;
  float x = 0;
  float y = 0;
  float z = 0;
  float yaw = 0;
  float vx = 0;
  float vy = 0;
  float yaw_speed = 0;
  float body_height = 0;
};

class StateReceiver {
 public:
  void start() {
    subscriber_.reset(
        new unitree::robot::ChannelSubscriber<
            unitree_go::msg::dds_::SportModeState_>("rt/sportmodestate"));
    subscriber_->InitChannel(
        std::bind(&StateReceiver::on_state, this, std::placeholders::_1), 32);
  }

  State get() {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

 private:
  void on_state(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
    State value;
    value.valid = true;
    value.monotonic_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (msg.position().size() >= 3) {
      value.x = msg.position()[0];
      value.y = msg.position()[1];
      value.z = msg.position()[2];
    }
    if (msg.velocity().size() >= 2) {
      value.vx = msg.velocity()[0];
      value.vy = msg.velocity()[1];
    }
    if (msg.imu_state().rpy().size() >= 3) {
      value.yaw = msg.imu_state().rpy()[2];
    }
    value.yaw_speed = msg.yaw_speed();
    value.body_height = msg.body_height();
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = value;
  }

  std::mutex mutex_;
  State state_;
  unitree::robot::ChannelSubscriberPtr<
      unitree_go::msg::dds_::SportModeState_>
      subscriber_;
};

std::string response(int rc, const State& state) {
  std::ostringstream out;
  out << std::setprecision(9) << "OK " << rc << ' '
      << (state.valid ? 1 : 0) << ' ' << state.monotonic_ns << ' ' << state.x
      << ' ' << state.y << ' ' << state.z << ' ' << state.yaw << ' '
      << state.vx << ' ' << state.vy << ' ' << state.yaw_speed << ' '
      << state.body_height << '\n';
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: go2_replay_bridge NETWORK_INTERFACE SOCKET_PATH "
                 "MAX_VX MAX_VY MAX_VYAW WATCHDOG_MS LEASE_TIMEOUT_S\n";
    return 2;
  }

  const std::string interface_name(argv[1]);
  socket_path = argv[2];
  const float max_vx = std::stof(argv[3]);
  const float max_vy = std::stof(argv[4]);
  const float max_vyaw = std::stof(argv[5]);
  const int watchdog_ms = std::stoi(argv[6]);
  const float lease_timeout_s = std::stof(argv[7]);
  if (max_vx <= 0 || max_vy <= 0 || max_vyaw <= 0 || watchdog_ms <= 0 ||
      lease_timeout_s <= 0) {
    std::cerr << "limits, watchdog, and lease timeout must be positive\n";
    return 2;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  std::atexit(remove_socket);

  unitree::robot::ChannelFactory::Instance()->Init(0, interface_name);
  unitree::robot::go2::SportClient client(/*enableLease=*/true);
  client.SetTimeout(lease_timeout_s);
  client.Init();
  std::cout << "waiting for SportClient lease" << std::endl;
  client.WaitLeaseApplied();
  std::cout << "SportClient lease applied" << std::endl;

  StateReceiver receiver;
  receiver.start();

  ::unlink(socket_path.c_str());
  const int server_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::cerr << "socket failed: " << std::strerror(errno) << '\n';
    return 1;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path)) {
    std::cerr << "socket path is too long\n";
    return 1;
  }
  std::strncpy(address.sun_path, socket_path.c_str(),
               sizeof(address.sun_path) - 1);
  if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0 ||
      ::listen(server_fd, 4) != 0) {
    std::cerr << "socket bind/listen failed: " << std::strerror(errno) << '\n';
    return 1;
  }
  const int flags = ::fcntl(server_fd, F_GETFL, 0);
  ::fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

  float vx = 0;
  float vy = 0;
  float vyaw = 0;
  bool moving = false;
  auto last_command = std::chrono::steady_clock::now();
  auto last_stream = last_command;
  uint64_t sequence = 0;
  int motion_fault = 0;
  constexpr int watchdog_fault = -4;
  log_motion("limits", "startup", sequence, 0, max_vx, max_vy, max_vyaw,
             monotonic_ns());

  auto stop = [&](const char* source) {
    const auto started_ns = monotonic_ns();
    const int rc = client.StopMove();
    log_motion("stop", source, sequence, rc, 0, 0, 0, started_ns);
    moving = false;
    vx = vy = vyaw = 0;
    return rc;
  };

  while (running.load()) {
    const int connection = ::accept(server_fd, nullptr, nullptr);
    if (connection >= 0) {
      timeval timeout{};
      timeout.tv_sec = 0;
      timeout.tv_usec = 20000;
      ::setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout));
      std::string input;
      char buffer[256];
      const auto receive_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
      while (input.find('\n') == std::string::npos && input.size() < 256 &&
             std::chrono::steady_clock::now() < receive_deadline) {
        const ssize_t count = ::read(connection, buffer, sizeof(buffer));
        if (count <= 0) break;
        input.append(buffer, buffer + count);
      }

      std::istringstream parser(input);
      std::string operation;
      parser >> operation;
      int rc = 0;
      if (operation == "MOVE") {
        ++sequence;
        float requested_vx = 0;
        float requested_vy = 0;
        float requested_vyaw = 0;
        if (!(parser >> requested_vx >> requested_vy >> requested_vyaw) ||
            !std::isfinite(requested_vx) || !std::isfinite(requested_vy) ||
            !std::isfinite(requested_vyaw)) {
          stop("invalid_command");
          rc = -2;
        } else if (motion_fault != 0) {
          rc = motion_fault;
          log_motion("move_rejected", "latched_fault", sequence, rc,
                     requested_vx, requested_vy, requested_vyaw, monotonic_ns());
        } else if (std::abs(requested_vx) > max_vx ||
                   std::abs(requested_vy) > max_vy ||
                   std::abs(requested_vyaw) > max_vyaw) {
          rc = motion_fault = -5;
          log_motion("move_rejected", "velocity_limits", sequence, rc,
                     requested_vx, requested_vy, requested_vyaw, monotonic_ns());
          stop("velocity_limits");
        } else {
          log_motion("request", "command", sequence, 0, requested_vx,
                     requested_vy, requested_vyaw, monotonic_ns());
          vx = requested_vx;
          vy = requested_vy;
          vyaw = requested_vyaw;
          const auto started_ns = monotonic_ns();
          rc = client.Move(vx, vy, vyaw);
          log_motion("move", "command", sequence, rc, vx, vy, vyaw, started_ns);
          moving = rc == 0;
          if (rc != 0) {
            motion_fault = rc;
            stop("move_failure");
          }
          last_command = std::chrono::steady_clock::now();
          last_stream = last_command;
        }
      } else if (operation == "STOP") {
        rc = stop("command");
        last_command = std::chrono::steady_clock::now();
      } else if (operation == "STAND_UP") {
        stop("stand_up");
        rc = client.StandUp();
        last_command = std::chrono::steady_clock::now();
      } else if (operation == "STAND_DOWN") {
        stop("stand_down");
        rc = client.StandDown();
        last_command = std::chrono::steady_clock::now();
      } else if (operation == "STATE") {
        rc = motion_fault;
      } else if (operation == "QUIT") {
        rc = stop("quit");
        running.store(false);
      } else {
        rc = -3;
      }

      const std::string output = response(rc, receiver.get());
      ::write(connection, output.data(), output.size());
      ::close(connection);
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
      std::cerr << "accept failed: " << std::strerror(errno) << '\n';
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto idle_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_command)
            .count();
    const auto stream_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stream)
            .count();
    if (moving && idle_ms > watchdog_ms) {
      std::cerr << "watchdog timeout; StopMove" << std::endl;
      motion_fault = watchdog_fault;
      stop("watchdog");
    } else if (moving && stream_ms >= 20) {
      const auto started_ns = monotonic_ns();
      const int rc = client.Move(vx, vy, vyaw);
      log_motion("move", "refresh", sequence, rc, vx, vy, vyaw, started_ns);
      if (rc != 0) {
        motion_fault = rc;
        stop("refresh_failure");
      }
      last_stream = std::chrono::steady_clock::now();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  stop("shutdown");
  ::close(server_fd);
  remove_socket();
  return 0;
}
