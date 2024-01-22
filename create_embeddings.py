import argparse
import pickle
import requests
import xmltodict
from app_config import AppConfig

from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


from langchain_community.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup
from site_loader import crawler_site


embedding = OpenAIEmbeddings(openai_api_key=AppConfig.getOpenAIKey())

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding website content')
    parser.add_argument('-s', '--sitemap', type=str, required=False,
            help='URL to your sitemap.xml', default='https://www.electrolux.vn/sitemap.xml')
    parser.add_argument('-f', '--filter', type=str, required=False,
            help='Text which needs to be included in all URLs which should be considered',
            default='https://www.electrolux.vn/blog')
    args = parser.parse_args()

    # r = requests.get(args.sitemap)
    # xml = r.text
    # raw = xmltodict.parse(xml)

    pages = []
    # for info in raw['urlset']['url']:
    #     # info example: {'loc': 'https://www.paepper.com/...', 'lastmod': '2021-12-28'}
    #     url = info['loc']
    #     if args.filter in url:
    #         pages.append({'text': extract_text_from(url), 'source': url})

    web_path=args.sitemap
    filter_urls= [
        "https://www.electrolux.vn/blog/cach-ve-sinh-tham-trai-san/",
        "https://www.electrolux.vn/blog/cach-giat-quan-ao-den-khong-bi-phai-mau/",
        "https://www.electrolux.vn/blog/vi-sao-tu-lanh-dong-da-cach-khac-phuc/",
        "https://www.electrolux.vn/blog/kich-thuoc-tu-lanh-pho-bien-hien-nay/",
        "https://www.electrolux.vn/blog/cach-tay-ao-trang-bi-o-vang/",
        "https://www.electrolux.vn/blog/cach-tay-son-tren-quan-ao/"
    ]
    site_contents = crawler_site(web_path, filter_urls)

    for info in site_contents:
        pages.append({'text': info["content"], 'source': info["url"]})

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    vectorindex_openai = FAISS.from_texts(docs, embedding, metadatas=metadatas)
    # with open("faiss_store.pkl", "wb") as f:
    #     pickle.dump(store, f)

    vectorindex_openai.save_local("electrolux_vn_store")
