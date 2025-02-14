from enum import Enum
from random import choice
from typing import List, Dict, Tuple
from urllib.parse import urlencode, quote
from Crypto.Cipher import AES
from base64 import b64encode
import datetime
import requests
import json
import pandas as pd
from pathlib import Path
# import sys
# sys.path.append('..')
# os.chdir(os.path.dirname(__file__))



# Keyword_Index_Dic = {'股市':'ZZQZ','股票':'ZZQZ','a股':'SZZS',
#     '上证':'SZZS','上证指数':'SZZS',
#     '基金':'BK0536','牛市':'上证','熊市':'上证',
#     '港股':'HSI','恒生指数':'HSI','恒生科技指数':'HSTECH',
#     '美股行情':'IXIC','道琼斯指数':'DQS','纳斯达克指数':'IXIC','中概股':'PGJ',
#     '上证50':'SZ50','沪深300':'HS300',
#     '中证500':'ZZ500', '创业板指':'399006','科创50':'KC50',
#     '螺纹钢':'RB0'}

# Index_Plt_Dic = {'SZZS':['股市','股票','a股','上证','上证指数'],
#                  'ZZQZ':['股市','股票','a股','上证','上证指数'],
#                  '上证':['牛市','熊市'],
#                  'HSI':['港股','恒生指数'],
#                  'HSTECH':['恒生科技指数'],
#                  'IXIC':['美股行情','道琼斯指数','纳斯达克指数'],
#                 #  'DQS':['美股行情','道琼斯指数','纳斯达克指数'],
#                 #  'PGJ':['中概股'],
#                  'SZ50':['股市', '上证50'],
#                  'HS300':['股市', '沪深300'],
#                  'ZZ500':['股市', '中证500'],
#                  '399006':['股市', '创业板指'],
#                  'KC50':['股市', '科创50'],
#                  'RB0':['螺纹钢']}

Basic_Cookie_Path = Path(Path(__file__).parent, '.cooks').absolute()
Search_Name_Path = Path(Path(__file__).parent, 'search_names.csv').absolute()

class ErrorCode(int, Enum):
    UNKNOWN = 10002
    NETWORK_ERROR = 10003

    # 百度指数
    NO_LOGIN = 20000
    KEYWORD_LIMITED = 20001
    REQUEST_LIMITED = 20002
    CHECK_KEYWORD_LIMITED = 20003

    # 百度的登录
    GET_QR_FAIL = 20010
    LOGIN_FAIL = 20011
    INDEX_LOGIN_FAIL = 20012


CODE_MSG_MAP = {
    ErrorCode.NO_LOGIN: 'cookies失效, 请重新获取cookies',
    ErrorCode.UNKNOWN: '未知错误',
    ErrorCode.NETWORK_ERROR: '网络错误',
    ErrorCode.KEYWORD_LIMITED: ('关键词最多传递5个, '
                                '可以使用`from qdata.baidu_index.common import split_keywords`,'
                                '对关键词进行切分'),
    ErrorCode.REQUEST_LIMITED: "该账号请求过于频繁, 请降低请求频率",
    ErrorCode.CHECK_KEYWORD_LIMITED: "最多传入15个关键词",
    ErrorCode.GET_QR_FAIL: "获取二维码失败",
    ErrorCode.LOGIN_FAIL: "百度登录失败",
    ErrorCode.INDEX_LOGIN_FAIL: "百度指数登录失败"
}


class GopupError(Exception):
    def __init__(self, code: ErrorCode, info: str = ""):
        self.code = code
        self.msg = CODE_MSG_MAP.get(code) + (info and f", {info}")

    def __str__(self):
        return repr(f"ERROR-{self.code}: {self.msg}")
    

HEADERS = {
    'Host': 'index.baidu.com',
    'Connection': 'keep-alive',
    'X-Requested-With': 'XMLHttpRequest',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36',
}

ALL_KIND = ['all', 'pc', 'wise']

Trys_Keywords = ['海关','数据','土地','安防','皮带','潘多拉','阿基米德','目的地','奥迪a6','编辑器','谭咏麟','企鹅体育','美丽的神话','法华寺','山田凉介','艾格拉斯','梅洛庞蒂']


def decrypt_func(key: str, data: str) -> List[str]:
    """
        数据解密方法
    """
    a = key
    i = data
    n = {}
    s = []
    for o in range(len(a)//2):
        n[a[o]] = a[len(a)//2 + o]
    for r in range(len(data)):
        s.append(n[i[r]])
    return ''.join(s).split(',')


def http_get(url: str, cookies: str, cipher_text: str = "") -> str:
    """
        发送get请求, 程序中所有的get都是调这个方法
        如果想使用多cookies抓取, 和请求重试功能
        在这自己添加
    """
    _headers = HEADERS.copy()
    _headers['Cookie'] = cookies
    if cipher_text:
        _headers["Cipher-Text"] = cipher_text
    try:
        response = requests.get(url, headers=_headers, timeout=30)
    except requests.Timeout:
        raise GopupError(ErrorCode.NETWORK_ERROR)
    if response.status_code != 200:
        raise GopupError(ErrorCode.NETWORK_ERROR)
    return response.text


def get_cipher_text(keyword: str) -> str:
    byte_list = [
        b"\x00", b"\x01", b"\x02", b"\x03", b"\x04", b"\x05", b"\x06", b"\x07",
        b"\x08", b"\x09", b"\x0a", b"\x0b", b"\x0c", b"\x0d", b"\x0e", b"\x0f",
        b"\x10"
    ]
    # 这个数是从acs-2057.js里写死的，但这个脚本请求时代时间戳，不确定是不是一个动态变化的脚本
    start_time = 1652338834776
    end_time = int(datetime.datetime.now().timestamp()*1000)

    wait_encrypted_data = {
        "ua": HEADERS["User-Agent"],
        "url": quote(f"https://index.baidu.com/v2/main/index.html#/trend/{keyword}?words={keyword}"),
        "platform": "MacIntel",
        "clientTs": end_time,
        "version": "2.1.0"
    }
    password = b"yyqmyasygcwaiyaa"
    iv = b"1234567887654321"
    aes = AES.new(password, AES.MODE_CBC, iv)
    wait_encrypted_str = json.dumps(wait_encrypted_data).encode()
    filled_count = 16 - len(wait_encrypted_str) % 16
    wait_encrypted_str += byte_list[filled_count] * filled_count
    encrypted_str = aes.encrypt(wait_encrypted_str)
    cipher_text = f"{start_time}_{end_time}_{b64encode(encrypted_str).decode()}"
    return cipher_text


def get_encrypt_json(
    *,
    start_date: str,
    end_date: str,
    keywords: List[List[str]],
    type: str,
    area: int,
    cookies: str
) -> Dict:
    pre_url_map = {
        'search': 'http://index.baidu.com/api/SearchApi/index?',
        'live': 'http://index.baidu.com/api/LiveApi/getLive?',
        'news': 'http://index.baidu.com/api/NewsApi/getNewsIndex?',
        'feed': 'http://index.baidu.com/api/FeedSearchApi/getFeedIndex?'
    }

    pre_url = pre_url_map[type]
    word_list = [
        [{'name': keyword, 'wordType': 1} for keyword in keyword_list]
        for keyword_list in keywords
    ]
    if type == 'live':
        request_args = {
            'word': json.dumps(word_list),
            'region': area
        }
    else:
        request_args = {
            'word': json.dumps(word_list),
            'startDate': start_date,
            'endDate': end_date,
            'area': area
        }
    url = pre_url + urlencode(request_args)
    cipher_text = get_cipher_text(keywords[0][0])
    html = http_get(url, cookies, cipher_text=cipher_text)
    datas = json.loads(html)
    if datas['status'] == 10000:
        raise GopupError(ErrorCode.NO_LOGIN)
    if datas["status"] == 10001:
        raise GopupError(ErrorCode.REQUEST_LIMITED)
    if datas['status'] != 0:
        raise GopupError(ErrorCode.UNKNOWN, str(datas))
    return datas


def get_key(uniqid: str, cookies: str) -> str:
    url = 'http://index.baidu.com/Interface/api/ptbk?uniqid=%s' % uniqid
    html = http_get(url, cookies)
    datas = json.loads(html)
    key = datas['data']
    return key


def format_data(data: Dict, kind: str):
    """
        格式化堆在一起的数据
    """
    keyword = str(data['word'])
    start_date = datetime.datetime.strptime(data['all']['startDate'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(data['all']['endDate'], '%Y-%m-%d')
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += datetime.timedelta(days=1)

    # for kind in ALL_KIND:
    index_datas = data[kind]['data']
    for i, cur_date in enumerate(date_list):
        try:
            index_data = index_datas[i]
        except IndexError:
            index_data = ''
        formated_data = {
            'keyword': [keyword_info['name'] for keyword_info in json.loads(keyword.replace('\'', '"'))][0],
            'type': kind,
            'date': cur_date.strftime('%Y-%m-%d'),
            'index': index_data if index_data else '0'
        }
        yield formated_data


def format_data_feed(data: Dict):
    keyword = str(data['key'])
    start_date = datetime.datetime.strptime(data['startDate'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(data['endDate'], '%Y-%m-%d')
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += datetime.timedelta(days=1)

    index_datas = data['data']
    for i, cur_date in enumerate(date_list):
        try:
            index_data = index_datas[i]
        except IndexError:
            index_data = ''
        formated_data = {
            'keyword': [keyword_info['name'] for keyword_info in json.loads(keyword.replace('\'', '"'))][0],
            'date': cur_date.strftime('%Y-%m-%d'),
            'index': index_data if index_data else '0'
        }
        yield formated_data


def format_data_new(data: Dict):
    keyword = str(data['key'])
    start_date = datetime.datetime.strptime(data['startDate'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(data['endDate'], '%Y-%m-%d')
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += datetime.timedelta(days=1)

    index_datas = data['data']
    for i, cur_date in enumerate(date_list):
        try:
            index_data = index_datas[i]
        except IndexError:
            index_data = ''
        formated_data = {
            'keyword': [keyword_info['name'] for keyword_info in json.loads(keyword.replace('\'', '"'))][0],
            'date': cur_date.strftime('%Y-%m-%d'),
            'index': index_data if index_data else '0'
        }
        yield formated_data

def baidu_search_index(word, start_date, end_date, cookie, type="all"):
    ''' 百度搜索数据 '''
    try:
        if isinstance(word,str):
            keywords_list = [[word]]
        elif isinstance(word,(List,Tuple)):
            keywords_list = [[x] for x in word]
        encrypt_json = get_encrypt_json(
            start_date=start_date,
            end_date=end_date,
            keywords=keywords_list,
            type='search',
            area=0,
            cookies=cookie
        )

        encrypt_datas = encrypt_json['data']['userIndexes']
        uniqid = encrypt_json['data']['uniqid']

        result = []
        key = get_key(uniqid, cookie)
        for encrypt_data in encrypt_datas:
            encrypt_data[type]['data'] = decrypt_func(key, encrypt_data[type]['data'])

            for formated_data in format_data(encrypt_data, kind=type):
                result.append(formated_data)
                # yield formated_data

        data_df = pd.DataFrame(result)
        data_df.index = pd.to_datetime(data_df["date"])
        del data_df["date"]
        return data_df
    except Exception as e:
        print(e)
        return None
    
def get_time_lst(tml:str)->List:
    st, ed = tml.split('|')
    sth, edh = int(st.split()[1][:2]), int(ed.split()[1][:2])
    stl = ['{} {:02d}:00:00'.format(st[:10],i) for i in range(sth,24)]
    edl = ['{} {:02d}:00:00'.format(ed[:10],i) for i in range(edh+1)]
    return stl+edl

def baidu_search_hour_index(word, cookie, kind="all"):
    try:
        if isinstance(word,str):
            keywords_list = [[word]]
        elif isinstance(word,(List,Tuple)):
            keywords_list = [[x] for x in word]
        encrypt_json = get_encrypt_json(
            start_date='000',
            end_date='000',
            keywords=keywords_list,
            type='live',
            area=0,
            cookies=cookie
        )

        encrypt_datas = encrypt_json['data']['result']
        uniqid = encrypt_json['data']['uniqid']
        result = []
        key = get_key(uniqid, cookie)
        for encrypt_data in encrypt_datas:
            encrypt_all = decrypt_func(key, encrypt_data['index'][0]['_'+kind])
            time_lst = get_time_lst(encrypt_data['index'][0]['period'])
            cname = encrypt_data['key'][0]['name']
            result.append(pd.DataFrame({cname:encrypt_all},index=time_lst))
        return pd.concat(result, axis=1)
    except Exception as e:
        print(e)
        return None
    
def baidu_info_index(word, start_date, end_date, cookie):
    ''' 百度资讯指数 '''
    try:
        keywords_list = [[word]]
        encrypt_json = get_encrypt_json(
            start_date=start_date,
            end_date=end_date,
            keywords=keywords_list,
            type='feed',
            area=0,
            cookies=cookie
        )

        encrypt_datas = encrypt_json['data']['index']
        uniqid = encrypt_json['data']['uniqid']

        result = []
        key = get_key(uniqid, cookie)
        for encrypt_data in encrypt_datas:
            encrypt_data['data'] = decrypt_func(key, encrypt_data['data'])

            for formated_data in format_data_feed(encrypt_data):
                result.append(formated_data)
                # yield formated_data

        data_df = pd.DataFrame(result)
        data_df.index = pd.to_datetime(data_df["date"])
        del data_df["date"]
        return data_df
    except Exception as e:
        return None

def baidu_media_index(word, start_date, end_date, cookie):
    ''' 百度媒体指数 '''
    try:
        keywords_list = [[word]]
        encrypt_json = get_encrypt_json(
            start_date=start_date,
            end_date=end_date,
            keywords=keywords_list,
            type='news',
            area=0,
            cookies=cookie
        )

        encrypt_datas = encrypt_json['data']['index']
        uniqid = encrypt_json['data']['uniqid']

        result = []
        key = get_key(uniqid, cookie)
        for encrypt_data in encrypt_datas:
            encrypt_data['data'] = decrypt_func(key, encrypt_data['data'])

            for formated_data in format_data_new(encrypt_data):
                result.append(formated_data)
                # yield formated_data

        data_df = pd.DataFrame(result)
        data_df.index = pd.to_datetime(data_df["date"])
        del data_df["date"]
        return data_df
    except Exception as e:
        return None
    

def choose_cookie():
    """ 选择适当的cookie """
    with open(Basic_Cookie_Path,'r', encoding='utf8') as f:
        cookf = f.readlines()
    for r in cookf:
        word = choice(Trys_Keywords)
        m_d = '{:02d}-{:02d}'.format(choice(range(1,5)),choice(range(1,29)))
        bs = baidu_search_index(word,'2024-01-01','2024-'+m_d,r.strip())
        if isinstance(bs, pd.DataFrame):
            return r.strip()
    raise GopupError(ErrorCode.REQUEST_LIMITED)
    # return None


if __name__=='__main__':
    # print(Search_Name_Path)
    print(choose_cookie())
    pass