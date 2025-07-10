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
import pickle
from utils.data import *

def radians_to_degrees(radians):
    degrees = radians * (180 / 3.141592653589793)
    return degrees

path = "/home/checkpoints/test_27052025/viz_m20_0_0_multiagent/"
max_num_road_pts_per_polyline = 100
files = os.listdir(path)

for e, file in tqdm(enumerate(files)):
    with open(os.path.join(path, file), 'r') as f:
        data = json.load(f)

    name = data['name'].split('_physics')[0]
    after_name = data['name'].split('_physics')[1]
    split_list = after_name.split('_')
    veh_veh_tilting = split_list[0]
    veh_edge_tilting = split_list[1]
    goal_tilting = split_list[-1].split('.')[0]

    roads_data = data['roads']
    num_roads = len(roads_data)
    print(f"the number of roads is {num_roads}")
    final_roads = []
    final_road_types = []
    for n in range(num_roads):
        curr_road_rawdat = roads_data[n]['geometry']
        if isinstance(curr_road_rawdat, dict):
            # for stop sign, repeat x/y coordinate along the point dimension
            final_roads.append(np.array((curr_road_rawdat['x'], curr_road_rawdat['y'], 1.0)).reshape(1, -1).repeat(max_num_road_pts_per_polyline, 0))
            # print(f"the {n} it, print curr_road_rawdat['x']: {curr_road_rawdat['x']}")
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
    agent_data = []
    agent_types = []
    agent_goals = []
    agent_rewards = []
    parked_agent_ids = [] # fade these out
    for n in range(len(agents_data)):
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]
        ag_position = np.column_stack((x_values, y_values))
        ag_heading = np.array(agents_data[n]['heading']).reshape((-1, 1))
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        if np.linalg.norm(ag_velocity, axis=-1).mean() < 0.05:
            parked_agent_ids.append(n)
        ag_existence = np.array(agents_data[n]['existence']).reshape((-1, 1))

        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        agent_type = get_object_type_onehot(agents_data[n]['type'])

        rewards = np.array(agents_data[n]['reward']) * ag_existence

        goal_position_x = agents_data[n]['goal_position']['x']
        goal_position_y = agents_data[n]['goal_position']['y']
        goal_position = np.repeat(np.array([goal_position_x, goal_position_y])[None, :], len(ag_position), 0)
        
        is_ctrlsim = agents_data[n]['is_ctrl_sim']
        if is_ctrlsim:
            focal_agent_idx = n

        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
        agent_goals.append(goal_position)
        agent_rewards.append(rewards)
    
    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    agent_goals = np.array(agent_goals)
    agent_rewards = np.array(agent_rewards)
    parked_agent_ids = np.array(parked_agent_ids)
    scene_id = data['enum']
    final_road_points = final_roads
    agent_states = agent_data
    goals = agent_goals
    # focal_agent_idx = data['focal_agent_idx']

    print(f"focal agent index is {focal_agent_idx} for scene {scene_id}")
    other_agent_idx = -1

    agent_color = 'pink'
    focal_color = 'lightseagreen'
    agent_alpha = 0.25
    focal_alpha = 1.0
    
    focal_idx_veh_veh_rewards = agent_rewards[focal_agent_idx, :, 6]
    other_idx_veh_veh_rewards = agent_rewards[other_agent_idx, :, 6]
    # if there exists a timestep where the focal and other collide on the same timestep, take the first such timestep
    # otherwise, first_collision_timestep is set to 90.
    found_collision = False
    for ts in range(90):
        if focal_idx_veh_veh_rewards[ts] == 1 and other_idx_veh_veh_rewards[ts] == 1:
            first_collision_timestep = ts 
            found_collision = True
            break
    if not found_collision:
        first_collision_timestep = int(np.sum(agent_data[focal_agent_idx, :, -1]) - 1)
    
    if np.sum(focal_idx_veh_veh_rewards) == 0:
        print(f"Scene {scene_id} has no veh-veh collision, pass to next scene")
        continue

    first_collision_timestep = np.where(focal_idx_veh_veh_rewards == 1)[0][0]
    collision_idxs = list(np.where(agent_rewards[:, first_collision_timestep, 6] == 1)[0])
    collision_idxs.remove(focal_agent_idx)
    if len(collision_idxs) > 1:
         continue
    other_agent_idx = collision_idxs[0]

    intermediate_timesteps = [first_collision_timestep]
    cur_timestep = first_collision_timestep - 10
    while cur_timestep > 5:
        intermediate_timesteps.append(cur_timestep)
        cur_timestep -= 10
    
    num_intermediate_timesteps = len(intermediate_timesteps)
    intermediate_alphas = list((np.arange(num_intermediate_timesteps - 1) / (num_intermediate_timesteps - 1)) * 0.75 + 0.25)
    intermediate_alphas.append(1.0)
    intermediate_timesteps = intermediate_timesteps[::-1]

    intermediate_alphas.insert(0, 0.25)
    intermediate_timesteps.insert(0, 0)
    
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
    
    coordinates = agent_states[:, :, :2]
    coordinates_mask = agent_states[:, :, -1].astype(bool).copy()
    # mask out trajectory after final keyframe
    coordinates_mask[:, first_collision_timestep:] = False
    unmodified_coordinates_mask = agent_states[:, :, -1].astype(bool)
    
    for a in range(len(coordinates)):
        if a == focal_agent_idx:
            x_min = np.min(coordinates[a, :, 0][unmodified_coordinates_mask[a]]) - 25
            x_max = np.max(coordinates[a, :, 0][unmodified_coordinates_mask[a]]) + 25
            y_min = np.min(coordinates[a, :, 1][unmodified_coordinates_mask[a]]) - 25
            y_max = np.max(coordinates[a, :, 1][unmodified_coordinates_mask[a]]) + 25

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
    
    for a in range(len(coordinates)):
        if a == focal_agent_idx:
            color = focal_color
            alpha = focal_alpha
            zord = 3
            zord_evolve=5
        elif a == other_agent_idx:
            color = agent_color 
            alpha = focal_alpha 
            zord = 2
            zord_evolve=4
        else:
            color = 'bisque'
            alpha = agent_alpha
            zord = 2
            zord_evolve=4
        if a == focal_agent_idx or a == other_agent_idx:
            plt.plot(coordinates[a, :, 0][coordinates_mask[a]], coordinates[a, :, 1][coordinates_mask[a]], color=color, linewidth=0.75, zorder=zord, alpha=alpha)
        else:
            length = agent_states[a, first_collision_timestep, 5]
            width = agent_states[a, first_collision_timestep, 6]
            bbox_x_min = coordinates[a, first_collision_timestep, 0] - width / 2
            bbox_y_min = coordinates[a, first_collision_timestep, 1] - length / 2
            lw = (0.35) / ((x_max - x_min) / 140)
            rectangle = mpatches.FancyBboxPatch((bbox_x_min, bbox_y_min),
                                        width, length, ec="black", fc=color, linewidth=lw, alpha=0.25,
                                        boxstyle=mpatches.BoxStyle("Round", pad=0.3), zorder=4)
            
            t = transforms.Affine2D().rotate_deg_around(coordinates[a, first_collision_timestep, 0], coordinates[a, first_collision_timestep, 1], radians_to_degrees(agent_states[a, first_collision_timestep, 4]) - 90) + plt.gca().transData

            # Apply the transformation to the rectangle
            rectangle.set_transform(t)
            
            plt.gca().set_aspect('equal', adjustable='box')
            # Add the patch to the Axes
            plt.gca().add_patch(rectangle)

            
            heading_length = length / 2 + 1.5
            heading_angle_rad = agent_states[a, first_collision_timestep, 4]
            vehicle_center = coordinates[a, first_collision_timestep, :2]

            # Calculate end point of the heading line
            line_end_x = vehicle_center[0] + heading_length * math.cos(heading_angle_rad)
            line_end_y = vehicle_center[1] + heading_length * math.sin(heading_angle_rad)

            # Draw the heading line
            plt.plot([vehicle_center[0], line_end_x], [vehicle_center[1], line_end_y], color='black', zorder=6, alpha=0.25, linewidth=0.25 / ((x_max - x_min) / 140))
        
        if a == focal_agent_idx or a == other_agent_idx:
            for i in range(len(intermediate_timesteps)):
                if i == len(intermediate_timesteps) - 1:
                    if intermediate_timesteps[i] == int(np.sum(agent_data[focal_agent_idx, :, -1]) - 1):
                        ec = 'green'
                    else:
                        ec = 'red'
                    lw = (0.5) / ((x_max - x_min) / 140)
                else:
                    ec = 'black'
                    lw = (0.35) / ((x_max - x_min) / 140)
                
                length = agent_states[a, intermediate_timesteps[i], 5]
                width = agent_states[a, intermediate_timesteps[i], 6]
                bbox_x_min = coordinates[a, intermediate_timesteps[i], 0] - width / 2
                bbox_y_min = coordinates[a, intermediate_timesteps[i], 1] - length / 2
                
                rectangle = mpatches.FancyBboxPatch((bbox_x_min, bbox_y_min),
                                            width, length, ec=ec, fc=color, linewidth=lw, alpha=intermediate_alphas[i],
                                            boxstyle=mpatches.BoxStyle("Round", pad=0.3), zorder=zord_evolve)
                
                t = transforms.Affine2D().rotate_deg_around(coordinates[a, intermediate_timesteps[i], 0], coordinates[a, intermediate_timesteps[i], 1], radians_to_degrees(agent_states[a, intermediate_timesteps[i], 4]) - 90) + plt.gca().transData

                # Apply the transformation to the rectangle
                rectangle.set_transform(t)
                
                plt.gca().set_aspect('equal', adjustable='box')
                # Add the patch to the Axes
                plt.gca().add_patch(rectangle)

                if i == len(intermediate_timesteps) - 1:
                    heading_length = length / 2 + 1.5
                    heading_angle_rad = agent_states[a, first_collision_timestep, 4]
                    vehicle_center = coordinates[a, first_collision_timestep, :2]

                    # Calculate end point of the heading line
                    line_end_x = vehicle_center[0] + heading_length * math.cos(heading_angle_rad)
                    line_end_y = vehicle_center[1] + heading_length * math.sin(heading_angle_rad)

                    # Draw the heading line
                    plt.plot([vehicle_center[0], line_end_x], [vehicle_center[1], line_end_y], color='black', zorder=6, alpha=1.0, linewidth=0.25 / ((x_max - x_min) / 140))

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    for a in range(len(goals)):
        if a == focal_agent_idx:
            color = focal_color
            ec = 'teal'
        elif a == other_agent_idx:
            color = agent_color
            ec = 'violet'
        else:
            continue
        
        lw = 1/((x_max - x_min) / 70)
        plt.scatter(goals[a, 0, 0], goals[a, 0, 1], color=color, s=int(35 / ((x_max - x_min) / 70)), zorder=10, edgecolors='black', linewidths=lw)

    save_dir = "./images_collision_07062025"
    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.join(save_dir, f"sample_{scene_id}.png")
    plt.tight_layout()
    plt.savefig(image_name, dpi=500)
    plt.clf()
