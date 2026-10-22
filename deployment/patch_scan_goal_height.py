"""Opt-in manual-goal Z for the pinned ROS2 SCAN source; legacy default is unchanged."""
import argparse
from pathlib import Path


def patched(text):
    marker='fsm.rob2nav_use_goal_height'
    if marker in text:return text
    init='navi_mode_ = load_parameter<int>(node_, "fsm.navi_mode", -1);'
    goal='end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, rviz_goal_height_;'
    if text.count(init)!=1 or text.count(goal)!=1:raise ValueError('Unexpected SCAN manual-goal source; review before patching')
    text='#include <algorithm>\n#include <cmath>\n'+text
    text=text.replace(init,init+'\n    node_->declare_parameter<bool>("'+marker+'", false);')
    return text.replace(goal,
        'const double requested_height = msg->poses[0].pose.position.z;\n'
        '    const bool use_goal_height = node_->get_parameter("'+marker+'").as_bool();\n'
        '    const double goal_height = use_goal_height && std::isfinite(requested_height)\n'
        '        ? std::max(0.20, std::min(1.00, requested_height)) : rviz_goal_height_;\n'
        '    end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, goal_height;')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    target=a.source/'src/planner/plan_manage/src/scan_replan_fsm.cpp'
    before=target.read_text();after=patched(before)
    if after!=before:target.write_text(after)
    print('SCAN manual goal height: opt-in patch ready; rebuild scan_planner')
