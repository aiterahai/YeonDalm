"""
file name: cralwer.py

create time: 2023-03-29 15:36
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
import requests
import os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

base_path = "../ai/data/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.3'}

search_words = ['트와이스 나연',
                '트와이스 다현',
                '트와이스 채영',
                '트와이스 쯔위',
                '트와이스 지효',
                '트와이스 정연',
                '트와이스 미나',
                '트와이스 사나',
                '트와이스 모모',
                '방탄소년단 정국',
                '방탄소년단 지민',
                '방탄소년단 뷔',
                '방탄소년단 슈가',
                '방탄소년단 진',
                '방탄소년단 RM',
                '방탄소년단 제이홉',
                '에스파 윈터',
                '에스파 카리나',
                '에스파 지젤',
                '에스파 닝닝',
                '뉴진스 다니엘',
                '뉴진스 하니',
                '뉴진스 민지',
                '뉴진스 해린',
                '뉴진스 혜인',
                '아이브 이서',
                '아이브 장원영',
                '아이브 레이',
                '아이브 리즈',
                '아이브 가을',
                '아이브 안유진',
                '투모로우바이투게더 수빈',
                '투모로우바이투게더 휴닝카이',
                '투모로우바이투게더 범규',
                '투모로우바이투게더 태현',
                '투모로우바이투게더 연준',
                '르세라핌 김채원',
                '르세라핌 카즈하',
                '르세라핌 사쿠라',
                '르세라핌 홍은채',
                '르세라핌 허윤진',
                '있지 예지',
                '있지 리아',
                '있지 류진',
                '있지 채령',
                '있지 유나',
                '스테이씨 세이',
                '스테이씨 재이',
                '스테이씨 수민',
                '스테이씨 윤',
                '스테이씨 아이사',
                '스테이씨 시은',
                '엔하이픈 희승',
                '엔하이픈 제이',
                '엔하이픈 제이크',
                '엔하이픈 성훈',
                '엔하이픈 선우',
                '엔하이픈 정원',
                '엔하이픈 니키',
                '블랙핑크 리사',
                '블랙핑크 로제',
                '블랙핑크 지수',
                '블랙핑크 제니',
                '스트레이키즈 방찬',
                '스트레이키즈 리노',
                '스트레이키즈 창빈',
                '스트레이키즈 현진',
                '스트레이키즈 한',
                '스트레이키즈 필릭스',
                '스트레이키즈 승민',
                '스트레이키즈 아이엔',
                '빅뱅 탑',
                '빅뱅 태양',
                '빅뱅 대성',
                '빅뱅 지드래곤',
                '엑소 첸',
                '엑소 시우민',
                '엑소 찬열',
                '엑소 수호',
                '엑소 백현',
                '엑소 디오',
                '엑소 레이',
                '엑소 카이',
                '엑소 세훈',
                ]


def download_image(query, url, i):
    response = requests.get(url, headers=headers)
    path = base_path + query + '/image{}.jpg'.format(i)
    with open(path, 'wb') as f:
        f.write(response.content)


for query in search_words:
    try:
        os.mkdir(base_path + query + "/")
    except FileExistsError as e:
        continue
    url = 'https://www.google.com/search?q={}&tbm=isch'.format(query)
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.content, 'html.parser')
    image_elements = soup.select('img.rg_i')
    image_urls = [img['data-src'] for img in image_elements if 'data-src' in img.attrs]
    with ThreadPoolExecutor() as executor:
        for i, url in enumerate(image_urls):
            executor.submit(download_image, query, url, i)
            if i % 30 == 0:
                tqdm.write(f"Downloaded {i} images for {query}")
