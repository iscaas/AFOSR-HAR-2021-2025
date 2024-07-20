<p align="center">
  <img src="Figures/od_virat.png" width="1000"/>
</p> 

**OD-VIRAT** is an object detection benchmark developed for object detection tasks in challenging surveillance environments. This dataset is constructed from videos of the [VIRAT Ground 2.0 Dataset](https://viratdata.org/), comprising 329 surveillance video captured through stationary ground cameras mounted at significant heights (mostly atop buildings), spanning 11 distinct scenes.
<hr>

## OD-VIRAT Scenes
<table>
  <tr>
    <td><div><img src="Figures/scene1.gif" alt="Scene 1" width="500"></div></td>
    <td><div><img src="Figures/scene2.gif" alt="Scene 2" width="500"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 1</b></td>
  <td><p align="center"><b>Scene 2</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene3.gif" alt="Scene 3" width="500"></div></td>
    <td><div><img src="Figures/scene4.gif" alt="Scene 4" width="500"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 3</b></td>
  <td><p align="center"><b>Scene 4</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene5.gif" alt="Scene 5" width="500"></div></td>
    <td><div><img src="Figures/scene6.gif" alt="Scene 6" width="500"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 5</b></td>
  <td><p align="center"><b>Scene 6</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene7.gif" alt="Scene 7" width="500"></div></td>
    <td><div><img src="Figures/scene8.gif" alt="Scene 8" width="500"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 7</b></td>
  <td><p align="center"><b>Scene 8</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene9.gif" alt="Scene 9" width="500"></div></td>
    <td><div><img src="Figures/scene10.gif" alt="Scene 10" width="500"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 9</b></td>
  <td><p align="center"><b>Scene 10</b></td>
  </tr>
</table>

|  Splits    | Scene 1 | Scene 2 | Scene 3 | Scene 4 | Scene 5 | Scene 6 | Scene 7 | Scene 8 | Scene 9 | Scene 10 |
|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
|    Train   |    5%   |    5%   |    4%   |   12%   |   13%   |   15%   |   27%   |    6%   |    5%   |    8%    | 


> Data distribution across 10 scenes in train, validation, and test sets of **OD-Virat Large** dataset.

Scenes       | Train           | Validation     | Test           |
-------------|-----------------|----------------|----------------|
Scene 1      | ███ 5%          | ████ 7%        | 0%             |
Scene 2      | ███ 5%          | ██████████ 21% | 0%             |
Scene 3      | ██ 4%           | ████ 8%        | ███ 5%         | 
Scene 4      | ██████ 12%      | █████ 9%       | ██████████ 21% |
Scene 5      | ███████ 13%     | ███ 5%         | █████ 9%       | 
Scene 6      | ████████ 15%    | ██████ 12%     | █████████ 17%  | 
Scene 7      | ███████████ 27% | █████████ 18%  | ████████ 15%   |
Scene 8      | ████ 6%         | ████████ 16%   | ████████ 15%   | 
Scene 9      | ███ 5%          | ██ 3%          | ██ 4%          | 
Scene 10     | █████ 8%        | █ 1%           | ████████ 14%   |


> Data distribution across 10 scenes in train, validation, and test sets of **OD-Virat Tiny** dataset.

Scenes       | Train           | Validation     | Test           |
-------------|-----------------|----------------|----------------|
Scene 1      | ████ 6%         | ████ 7%        | 0%             |
Scene 2      | ███ 5%          | ██████████ 22% | 0%             |
Scene 3      | ██ 4%           | ████ 8%        | ███ 5%         | 
Scene 4      | ██████ 12%      | █████ 9%       | ██████████ 21% |
Scene 5      | ███████ 13%     | ███ 5%         | █████ 9%       | 
Scene 6      | ████████ 15%    | ██████ 12%     | █████████ 17%  | 
Scene 7      | ███████████ 27% | █████████ 18%  | ████████ 15%   |
Scene 8      | ████ 6%         | ████████ 16%   | ████████ 15%   | 
Scene 9      | ███ 5%          | ██ 3%          | ██ 4%          | 
Scene 10     | █████ 8%        | █ 1%           | ████████ 14%   |

<img src="Figures/bbox_per_frame_v2.png" width="800"/>


