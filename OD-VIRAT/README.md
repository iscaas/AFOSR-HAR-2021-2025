# OD-VIRAT: A Large-Scale Benchmark for Object Detection in Realistic Surveillance Environments

<!-- [Hayat Ullah](https://scholar.google.com.pk/citations?user=xnXPj0UAAAAJ&hl=en),  -->
<!-- [Arslan Munir](https://people.cs.ksu.edu/~amunir/) -->


[![paper](https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg)]()
<hr />

> **Abstract:**
>*will add paper abstract here...*

## Table of Contents
<!--ts-->
   * [News](#rocket-News)
   * [Overview](#overview)
   * [Visualization](#visualization-first-and-last-layer-spatio-temporal-modulator)
   * [Environment Setup](#environment-setup)
   * [Dataset Preparation](#dataset-preparation)
   * [Evaluation](#evaluation)
   * [Training](#training)
   * [Citation](#citation)
   * [Acknowledgements](#acknowledgements)
<!--te-->

**OD-VIRAT** is an object detection benchmark developed for object detection tasks in challenging surveillance environments. This dataset is constructed from videos of the [VIRAT Ground 2.0 Dataset](https://viratdata.org/), comprising 329 surveillance video captured through stationary ground cameras mounted at significant heights (mostly atop buildings), spanning 11 distinct scenes.
<hr>
<table class="small-font-table">
  <tr>
    <th rowspan="2">Dataset</th>
    <th colspan="3"># of Scenes</th>
    <th colspan="3"># of Videos</th>
    <th colspan="3"># of Frames</th>
    <th rowspan="2">Resolution</th>
  </tr>
  <tr>
    <td><b>train</b></td>
    <td><b>validation</b></td>
    <td><b>test</b></td>
    <td><b>train</b></td>
    <td><b>validation</b></td>
    <td><b>test</b></td>
    <td><b>train</b></td>
    <td><b>validation</b></td>
    <td><b>test</b></td>
  </tr>
  <tr>
    <td><b>OD-VIRAT Large</b></td>
    <td><p align="center">10</td>
    <td><p align="center">10</td>
    <td><p align="center">8</td>
    <td><p align="center">156</td>
    <td><p align="center">52</td>
    <td><p align="center">52</td>
    <td><p align="center">377686</td>
    <td><p align="center">137971</td>
    <td><p align="center">84339</td>
    <td><p align="center">(1920 &times 1080),(1280 &times 720)</td>
  </tr>
   <tr>
    <td><b>OD-VIRAT Tiny</b></td>
    <td><p align="center">10</td>
    <td><p align="center">10</td>
    <td><p align="center">8</td>
    <td><p align="center">156</td>
    <td><p align="center">52</td>
    <td><p align="center">52</td>
    <td><p align="center">12501</td>
    <td><p align="center">4573</td>
    <td><p align="center">2786</td>
    <td><span style="font-size: smaller;">(1920 &times 1080),(1280 &times 720)</span></td>
  </tr>
</table>
<hr>

## OD-VIRAT Scenes
<table>
  <tr>
    <td><div><img src="Figures/scene1.gif" alt="Scene 1" width="400"></div></td>
    <td><div><img src="Figures/scene2.gif" alt="Scene 2" width="400"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 1</b></td>
  <td><p align="center"><b>Scene 2</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene3.gif" alt="Scene 3" width="400"></div></td>
    <td><div><img src="Figures/scene4.gif" alt="Scene 4" width="400"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 3</b></td>
  <td><p align="center"><b>Scene 4</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene5.gif" alt="Scene 5" width="400"></div></td>
    <td><div><img src="Figures/scene6.gif" alt="Scene 6" width="400"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 5</b></td>
  <td><p align="center"><b>Scene 6</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene7.gif" alt="Scene 7" width="400"></div></td>
    <td><div><img src="Figures/scene8.gif" alt="Scene 8" width="400"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 7</b></td>
  <td><p align="center"><b>Scene 8</b></td>
  </tr>
  <tr>
    <td><div><img src="Figures/scene9.gif" alt="Scene 9" width="400"></div></td>
    <td><div><img src="Figures/scene10.gif" alt="Scene 10" width="400"></div></td>
  </tr>
  <tr>
  <td><p align="center"><b>Scene 9</b></td>
  <td><p align="center"><b>Scene 10</b></td>
  </tr>
</table>

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


