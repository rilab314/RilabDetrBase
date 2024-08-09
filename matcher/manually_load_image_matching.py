import numpy as np
import cv2
import math
import geopandas as gpd
import pandas as pd
import os
from glob import glob
from PIL import Image
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import LineString
from tqdm import tqdm


# 파일 이름에서 WebMercator 좌표 추출
def extract_coords(filename):
    parts = filename.split('_')[1].replace('.png', '').split(',')
    return list(map(float, parts))


def transform_geometry(geometry):
    if geometry is not None:
        # transform 함수를 수정하여 3D 좌표에서 z 값을 무시합니다.
        if geometry.geom_type == 'LineString':
            return LineString([transformer.transform(x, y) for x, y, *_ in geometry.coords])
        else:
            # 기타 지오메트리 유형에 대해서는 모든 좌표에서 z 값을 무시합니다.
            return transform(lambda x, y, z=0: transformer.transform(x, y), geometry)
    else:
        # geometry가 None일 경우, None을 반환
        return None


# 도로 데이터를 픽셀 좌표로 변환
def convert_geometry_to_pixels(geom):
    return [coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height) for x, y, *_ in np.array(geom.coords)]


# 좌표를 이미지 픽셀로 변환
def coords_to_pixels(x, y, x_min, y_max, x_max, y_min, width, height):
    x_pixel = int((width * (x - x_min) / (x_max - x_min)) * ratio + x_ad)
    y_pixel = int((height * (y - y_max) / (y_min - y_max)) * ratio + y_ad)
    return x_pixel, y_pixel


# 이미지에 도로 그리기 함수
def draw_roads(img, pixel_coords):
    for line_pixels in pixel_coords:
        prev_point = None
        if np.all(np.isnan(line_pixels)) or np.all(np.isnan(line_pixels)):
            continue
        for point in line_pixels:
            if prev_point is not None:
                cv2.line(img, prev_point, point, (0, 255, 255), 1)
            prev_point = point
    return img


def find_files(root_folder, file_name_to_find):
    found_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if os.path.join(subdir, file).endswith(file_name_to_find):
                found_files.append(os.path.join(subdir, file))
    return found_files


# Road all A2_LINK.shp files
# road_links_paths1 = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/(B110)정밀도로지도_서울특별시_강남구/*/HDMap_UTM52N_타원체고/A2_LINK.shp")
# road_links_paths2 = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/(B110)정밀도로지도_서울특별시_강남구/*/*/HDMap_UTM52N_타원체고/A2_LINK.shp")
# road_links_paths = road_links_paths1 + road_links_paths2

root_folder = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토지리/unzip'  # 여기에 원하는 폴더 경로를 입력하세요.
road_links_paths = find_files(root_folder, '/HDMap_UTM52N_타원체고/A2_LINK.shp')
road_links_paths.sort()
# road_links_paths=['/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/국토지리/unzip/(B110)정밀도로지도_서울특별시_강남구/서울특별시도_41호선/SEC005_헌릉로869_송파IC/HDMap_UTM52N_타원체고/A2_LINK.shp']
total_road_links = []
transformer = Transformer.from_crs('EPSG:32652', 'EPSG:3857', always_xy=True)
for r_l_path in tqdm(road_links_paths, desc="Transforming Road Datas"):
    road_links = gpd.read_file(r_l_path)
    road_links['transformed_geometry'] = road_links['geometry'].apply(transform_geometry)
    total_road_links.append(road_links['transformed_geometry'])



# 조정 변수 초기화
x_ad, y_ad = 0, 0
ratio = 1
image_num = 58


# songpa IC
# image_list = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam/gangnam_송파IC/1024x1024/box/*.png")
# image_list.sort()

# gangnam around dogok station
image_list = glob("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam/gangnam_around_dogok_station/1024x1024/box/*.png")
image_list.sort()



### 하나의 큰 이미지
# image_path = ("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam"
#               "/송파IC_14151418.846246604,4504815.067658295,14152615.563470695,4506231.5413387045_crop.png")
# # image_path = ("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam"
# #               "/송파_14150591.176549697,4503428.452192059,14152738.579313084,4505900.712325657_crop.png")
# # image_path = ("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam"
# #               "/gangnam_14140601.812415678,4507797.306089984,14144309.00828751,4511580.9389901_crop.png")
# image_path = ("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA/gangnam"
#               "/gangnamall_14138366.029338337,4502064.528968595,14152659.753627665,4513835.831324512_crop.png")
image_path = ("/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/NOAA"
              "/seoul_14134247.984439474,4506813.179350813,14146057.505309533,4517695.901252915_crop.png")

##
image_name = image_path.split('/')[-1]
x_min, y_min, x_max, y_max = extract_coords(image_name)
width, height = Image.open(image_path).size

image = cv2.imread(image_path)
for i in total_road_links:
    try:
        road_links['pixel_coords'] = i.apply(convert_geometry_to_pixels)
        image = draw_roads(image, road_links['pixel_coords'])
    except:
        print(i)
        continue

save_root = image_path.replace('_crop', '_crop_drew')
cv2.imwrite(save_root, image)
# cv2.imshow('Around_Road Map', image)
# cv2.imshow('Around_original', cv2.imread(image_path))


# 메인 루프
while True:
    ##### 작은 이미지들
    image_path = image_list[image_num]

    image_name = image_path.split('/')[-1]
    x_min, y_min, x_max, y_max = extract_coords(image_name)
    width, height = Image.open(image_path).size

    image = cv2.imread(image_path)
    for i in tqdm(total_road_links, desc="Drawing Road Datas"):
        try:
            road_links['pixel_coords'] = i.apply(convert_geometry_to_pixels)
            image = draw_roads(image, road_links['pixel_coords'])
        except:
            print(i)
            continue
    # cv2.imshow('Road Map', image)
    # cv2.imshow('original', cv2.imread(image_path))
    save_root = image_path.replace('box', 'box_drew')
    cv2.imwrite(save_root, image)
    print(f"save {save_root}")
    image_num += 1

    # key = cv2.waitKey(0)
    # if key == 27:
    #     break
    # elif key == ord('g'):
    #     image_num += 1
    # elif key == ord('f'):
    #     image_num -= 1
    # elif key == ord('d'):
    #     x_ad += 1
    # elif key == ord('a'):
    #     x_ad -= 1
    # elif key == ord('s'):
    #     y_ad += 1
    # elif key == ord('w'):
    #     y_ad -= 1
    # elif key == ord('q'):
    #     ratio += 0.001
    # elif key == ord('e'):
    #     ratio -= 0.001
    # print(f"x_ad: {x_ad}, y_ad: {y_ad}, ratio: {ratio}")

cv2.destroyAllWindows()
