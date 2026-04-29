# AVSO SAM3

SAM3 기반 occlusion-robust video object segmentation. 시각적으로 유사한 객체가 혼재하는 환경에서 detector-tracker 파이프라인을 통해 안정적인 추적을 수행합니다.

## LVOS Evaluation Results

<div align="center">
<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="3" align="center">LVOSv1</th>
      <th colspan="3" align="center">LVOSv2</th>
    </tr>
    <tr>
      <th></th>
      <th align="center">J&amp;F</th>
      <th align="center">J</th>
      <th align="center">F</th>
      <th align="center">J&amp;F</th>
      <th align="center">J</th>
      <th align="center">F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SAM3</td>
      <td align="center">84.0</td>
      <td align="center">78.5</td>
      <td align="center">89.4</td>
      <td align="center">88.5</td>
      <td align="center">84.7</td>
      <td align="center">92.3</td>
    </tr>
    <tr>
      <td><strong>Ours</strong></td>
      <td align="center"><strong>86.8</strong></td>
      <td align="center"><strong>81.2</strong></td>
      <td align="center"><strong>92.4</strong></td>
      <td align="center"><strong>89.2</strong></td>
      <td align="center"><strong>85.4</strong></td>
      <td align="center"><strong>93.1</strong></td>
    </tr>
  </tbody>
</table>
</div>
