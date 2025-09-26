""" This script is to generate the image with agent id 
"""
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import json
import math
from tqdm import tqdm 
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.patheffects as patheffects
import pickle
import random
import subprocess
from moviepy import ImageSequenceClip

PATH = r"C:\Users\cwang76\Downloads\viz_res\test"  # load path which should be a folder

frame_idx = 1    # which timestep we want to plot 
save_dir = './test22092025'  # output path 

_COLOR = {
    "vehicle": "pink",
    "pedestrian": "blue",
    "cyclist": "green"
}

def radians_to_degrees(radians):
    degrees = radians*(180/3.141592653589793)
    return degrees

def degrees_to_radians(degrees):
    radians = degrees*(3.141592653589793/180)
    return radians

def get_road_type_onehot(road_type):
    road_types = {"none": 0, "lane": 1, "road_line": 2, "road_edge": 3, "stop_sign": 4, "crosswalk": 5,
                      "speed_bump": 6, "other": 7}
    return np.eye(len(road_types))[road_types[road_type]]

def get_object_type_onehot(agent_type):
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]



max_num_road_pts_per_polyline = 100

files = os.listdir(PATH)

for e, file in enumerate(files):
    with open(os.path.join(PATH, file), 'r') as f:
        data = json.load(f)
    filename = file.split('.')[0]
    seed = file.split('_')[-1]
    seed = seed.split('.')[0]    
    name = filename.split('_')[3]
    roads_data = data['roads']
    num_roads = len(roads_data)
    final_roads = []
    final_road_types = []
    for n in range(num_roads):
        curr_road_rawdat = roads_data[n]['geometry']
        if isinstance(curr_road_rawdat, dict):
            # for stop sign, repeat x/y coordinate along the point dimension
            final_roads.append(np.array((curr_road_rawdat['x'], curr_road_rawdat['y'], 1.0)).reshape(1, -1).repeat(max_num_road_pts_per_polyline, 0))
            final_road_types.append(get_road_type_onehot(roads_data[n]['type']))
        else:
            # either we add points until we run out of points and append zeros
            # or we fill up with points until we reach max limit
            curr_road = []
            for p in range(len(curr_road_rawdat)):
                curr_road.append(np.array((curr_road_rawdat[p]['x'], curr_road_rawdat[p]['y'], 1.0)))
                if len(curr_road) == max_num_road_pts_per_polyline:
                    final_roads.append(np.array(curr_road))
                    curr_road = []
                    final_road_types.append(get_road_type_onehot(roads_data[n]['type']))
            if len(curr_road) < max_num_road_pts_per_polyline and len(curr_road) > 0:
                tmp_curr_road = np.zeros((max_num_road_pts_per_polyline, 3))
                tmp_curr_road[:len(curr_road)] = np.array(curr_road)
                final_roads.append(tmp_curr_road)
                final_road_types.append(get_road_type_onehot(roads_data[n]['type']))

    final_roads = np.array(final_roads)
    final_road_types = np.array(final_road_types)

    agents_data = data['objects']
    num_agents = len(agents_data)
    agents_id = range(num_agents)

    agent_data = []
    agent_types = []
    agent_goals = []
    agent_ctrlsim = []
    agents_color = []
    parked_agent_ids = [] # fade these out
    for n in range(len(agents_data)):
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]

        ag_existence = np.ones((len(ag_position), 1))
        # ag_existence = [int(agent_existence) for agent_existence in data['objects'][2]['valid']]
        # ag_existence = np.array(ag_existence).reshape((-1, 1))
        for j in range(len(ag_existence)):
            if x_values[j] < -5000 or y_values[j] < -5000:
                ag_existence[j] = 0

        agent_color = 'pink'
        
        ag_position = np.column_stack((x_values, y_values))
        ag_heading = np.array(agents_data[n]['heading']).reshape((-1, 1))
        ag_heading = degrees_to_radians(ag_heading)
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        if np.linalg.norm(ag_velocity, axis=-1).mean() < 0.05:
            parked_agent_ids.append(n)

        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        agent_type = get_object_type_onehot(agents_data[n]['type'])
        

        goal_position_x = agents_data[n]['position'][70]['x']
        goal_position_y = agents_data[n]['position'][70]['y']
        goal_position = np.repeat(np.array([goal_position_x, goal_position_y])[None, :], len(ag_position), 0)

        # num_colone of ag_state: 8 (if there is ag_existence)
        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
        agent_goals.append(goal_position)
        agents_color.append(agent_color)


    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    agent_goals = np.array(agent_goals)
    parked_agent_ids = np.array(parked_agent_ids)

    final_road_points = final_roads
    agent_states = agent_data
    goals = agent_goals

    agent_alpha = 0.5
    agent_zord = 2
    
    coordinates = agent_states[:, :, :2]
    coordinates_mask = agent_states[:, :, -1].astype(bool).copy()
    
    x_min_all = 100000
    y_min_all = 100000
    x_max_all = -100000
    y_max_all = -100000
    for a in range(len(coordinates)):
        xs = coordinates[a, :, 0][coordinates_mask[a]]
        ys = coordinates[a, :, 1][coordinates_mask[a]]
        if xs.size == 0 or ys.size == 0:
            continue
        x_min = np.min(coordinates[a, :, 0][coordinates_mask[a]]) - 25
        x_max = np.max(coordinates[a, :, 0][coordinates_mask[a]]) + 25
        y_min = np.min(coordinates[a, :, 1][coordinates_mask[a]]) - 25
        y_max = np.max(coordinates[a, :, 1][coordinates_mask[a]]) + 25
        if x_min < x_min_all:
            x_min_all = x_min 
        if y_min < y_min_all:
            y_min_all = y_min 
        if x_max > x_max_all:
            x_max_all = x_max
        if y_max > y_max_all:
            y_max_all = y_max

    x_min = x_min_all 
    y_min = y_min_all 
    x_max = x_max_all 
    y_max = y_max_all

    print(f"x min is {x_min}, x max is {x_max}, y min is {y_min}, y max is {y_max}")


    if (x_max - x_min) > (y_max - y_min):
        diff = (x_max - x_min) - (y_max - y_min)
        diff_side = diff / 2
        y_min -= diff_side 
        y_max += diff_side 
    else:
        diff = (y_max - y_min) - (x_max - x_min)
        diff_side = diff / 2
        x_min -= diff_side 
        x_max += diff_side 
    
    for r in range(len(final_road_points)):
        if final_road_types[r, 3] != 1:
            continue
        mask = final_road_points[r, :, 2].astype(bool)
        plt.plot(final_road_points[r, :, 0][mask], final_road_points[r, :, 1][mask], color='grey', linewidth=0.5)
    
    for r in range(len(final_road_points)):
        if final_road_types[r, 2] != 1 and final_road_types[r, 2] != 1:
            continue
        mask = final_road_points[r, :, 2].astype(bool)
        plt.plot(final_road_points[r, :, 0][mask], final_road_points[r, :, 1][mask], color='lightgray', linewidth=0.3)


    for a in range(len(coordinates)):
        color = agents_color[a]
        alpha = agent_alpha
        zord = agent_zord
        edgecolor = 'black'
        label = None
        if coordinates[a, frame_idx, 0] < -1000 or coordinates[a, frame_idx, 1] < -1000:
            continue
        
        # draw bounding boxes
        length = agent_states[a, frame_idx, 5] * 0.8
        width = agent_states[a, frame_idx, 6] * 0.8
        bbox_x_min = coordinates[a, frame_idx, 0] - width / 2
        bbox_y_min = coordinates[a, frame_idx, 1] - length / 2
        lw = (0.35) / ((x_max - x_min) / 140)
        rectangle = mpatches.FancyBboxPatch((bbox_x_min, bbox_y_min),
                                    width, length, ec=edgecolor, fc=color, linewidth=lw, alpha=alpha,
                                    boxstyle=mpatches.BoxStyle("Round", pad=0.3), zorder=4, label=label)
        
        t = transforms.Affine2D().rotate_deg_around(coordinates[a, frame_idx, 0], coordinates[a, frame_idx, 1], radians_to_degrees(agent_states[a, frame_idx, 4]) - 90) + plt.gca().transData

        # Apply the transformation to the rectangle
        rectangle.set_transform(t)
        
        plt.gca().set_aspect('equal', adjustable='box')
        # Add the patch to the Axes
        plt.gca().add_patch(rectangle)
        
        # heading_length = length / 2 + 1.5
        # heading_angle_rad = agent_states[a, frame_idx, 4]
        # vehicle_center = coordinates[a, frame_idx, :2]

        # # Calculate end point of the heading line
        # line_end_x = vehicle_center[0] + heading_length * math.cos(heading_angle_rad)
        # line_end_y = vehicle_center[1] + heading_length * math.sin(heading_angle_rad)

        # # Draw the heading line
        # plt.plot([vehicle_center[0], line_end_x], [vehicle_center[1], line_end_y], color='black', zorder=6, alpha=0.25, linewidth=0.25 / ((x_max - x_min) / 140))

        
        cx, cy = coordinates[a, frame_idx, :2]
        # draw the id for each agent
        txt = plt.text(cx, cy, str(a),
                    ha='center', va='center', fontsize=6, color='black', zorder=7)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    plt.show()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{name}_frame{frame_idx}.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"image saved in: {save_path}")
