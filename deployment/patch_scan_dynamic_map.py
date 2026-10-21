"""Opt-in occupancy ageing experiment for the pinned community SCAN checkout.

Default timeout is zero, preserving normal/static mapping. Simulation only:
unobserved occupancy expires to unknown, which SCAN does not treat as occupied.
This is unsuitable as a standalone safety map for unobserved real obstacles.
"""
import argparse
from pathlib import Path
import subprocess

PIN='d0b921c9b05a6d291d144d60882b2e0e88d2c0e0'
def patch(root):
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==PIN
    folder=root/'src/planner/plan_env'
    header=folder/'include/plan_env/grid_map.h';source=folder/'src/grid_map.cpp'
    def read(path):return path.read_bytes().decode().replace('\r\r\n','\n').replace('\r\n','\n')
    h=read(header);s=read(source)
    marker='rob2nav_occupancy_timeout_'
    if marker in h:
        assert marker in s
        return
    def replace(text,old,new):
        assert text.count(old)==1,old
        return text.replace(old,new)
    h=replace(h,'#include <queue>','#include <queue>\n#include <map>')
    h=replace(h,'  MappingData md_;','  MappingData md_;\n  double rob2nav_occupancy_timeout_ = 0.0;\n  std::map<int, std::pair<double, Eigen::Vector3i>> rob2nav_recent_hits_;')
    s=replace(s,'  load_parameter(node_, "grid_map.ground_height", mp_.ground_height_, 0.0);','  load_parameter(node_, "grid_map.ground_height", mp_.ground_height_, 0.0);\n  load_parameter(node_, "grid_map.rob2nav_occupancy_timeout", rob2nav_occupancy_timeout_, 0.0);')
    s=replace(s,'void GridMap::resetAllMapData()\n{','void GridMap::resetAllMapData()\n{\n  rob2nav_recent_hits_.clear();')
    s=replace(s,'  // update occupancy cached in queue','''  // Optional simulation experiment: ageing is based on received hit evidence.
  // Expiry removes inflation through the normal update path, never labels free.
  const double rob2nav_now = node_->get_clock()->now().seconds();
  if (rob2nav_occupancy_timeout_ > 0.0)
  {
    for (auto it = rob2nav_recent_hits_.begin(); it != rob2nav_recent_hits_.end(); )
    {
      const auto id = it->second.second;
      if (!isInMap(id) || toAddress(id) != it->first)
        it = rob2nav_recent_hits_.erase(it);
      else if (rob2nav_now - it->second.first > rob2nav_occupancy_timeout_)
      {
        applyOccupancyUpdate(id, mp_.clamp_min_log_ - mp_.unknown_flag_);
        it = rob2nav_recent_hits_.erase(it);
      }
      else ++it;
    }
  }

  // update occupancy cached in queue''')
    s=replace(s,'    md_.count_hit_[idx_ctns] = md_.count_hit_and_miss_[idx_ctns] = 0;','''    if (rob2nav_occupancy_timeout_ > 0.0 && md_.count_hit_[idx_ctns] > 0)
      rob2nav_recent_hits_[idx_ctns] = std::make_pair(rob2nav_now, idx);
    md_.count_hit_[idx_ctns] = md_.count_hit_and_miss_[idx_ctns] = 0;''')
    header.write_text(h);source.write_text(s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);a=p.parse_args();patch(a.checkout)
