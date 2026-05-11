# 서울시 청년·저소득·임차가구 밀집 분석 요약

## 분석 설정
- 입력 파일: `data\processed\legal_dong_risk_base_2024.csv`
- 분석 대상: 총인구수 0 포함 법정동 467개
- CI 기본 가중치: 청년 0.35, 1인가구 0.30, 기초수급 0.35
- 경계 데이터: `southkorea/seoul-maps` JUSO 2015 GeoJSON (`data\raw\seoul_neighborhoods_geo_simple_2015.geojson`)

## 주요 결과 TOP 10
| priority_rank | 시군구명 | 법정동명 | concentration_index | cluster_label | lisa_cluster |
| --- | --- | --- | --- | --- | --- |
| 1 | 용산구 | 갈월동 | 0.7531 | 저소득 밀집형 | Not significant |
| 2 | 용산구 | 동자동 | 0.7525 | 저소득 밀집형 | Not significant |
| 3 | 용산구 | 남영동 | 0.7504 | 저소득 밀집형 | Not significant |
| 4 | 광진구 | 화양동 | 0.6859 | 청년 밀집형 | Not significant |
| 5 | 종로구 | 청진동 | 0.667 | 저소득 밀집형 | Not significant |
| 6 | 종로구 | 경운동 | 0.6543 | 저소득 밀집형 | HH |
| 7 | 종로구 | 관철동 | 0.6526 | 저소득 밀집형 | HL |
| 8 | 종로구 | 낙원동 | 0.6524 | 저소득 밀집형 | HH |
| 9 | 종로구 | 관수동 | 0.6524 | 저소득 밀집형 | HH |
| 10 | 종로구 | 원남동 | 0.6521 | 저소득 밀집형 | Not significant |

## K-means 군집 요약
| cluster_id | cluster_label | 법정동수 | 청년비율 | 1인가구비율 | 기초수급가구비율 | concentration_index |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 복합 밀집형 | 271 | 0.2761 | 0.3684 | 0.0627 | 0.3508 |
| 1 | 일반형 | 27 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 저소득 밀집형 | 34 | 0.2756 | 0.6551 | 0.2266 | 0.6206 |
| 0 | 청년 밀집형 | 135 | 0.4255 | 0.6052 | 0.0535 | 0.5126 |

## 공간 자기상관 요약
- Moran's I: 0.4176
- Queen 이웃 링크 수: 2,700
- 이웃이 없는 법정동 수: 1
- LISA permutation 수: 999

| lisa_cluster | 법정동수 |
| --- | --- |
| Not significant | 370 |
| LL | 44 |
| HH | 43 |
| HL | 8 |
| LH | 2 |

## 산출물
- `output\seoul_legal_dong_concentration.csv`
- `output\seoul_legal_dong_ci_rank.csv`
- `output\seoul_legal_dong_clusters.csv`
- `output\seoul_priority_top30.csv`
- `output\figures`
