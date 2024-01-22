from langchain_community.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup

class SiteContent:
    url: str
    page_content: str

    def __init__(self, url: str, page_content: str):
        self.url = url
        self.page_content = page_content


def get_blog_content(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    text_elements = content.find("div", {"class": "textimageblock"})
    if text_elements is not None:
        text = text_elements.get_text()
        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)
    else:
        return ''

def crawler_site(web_path: str, filter_urls: list[str]=[]) -> list[dict]:
    contents = [];
    sitemap_loader = SitemapLoader(
        web_path=web_path,
        filter_urls=filter_urls,
        parsing_function=get_blog_content
        )
    
    docs = sitemap_loader.load()
    for x in docs:
        contents.append(dict(
            url =x.metadata['source'],
            content =x.page_content
        ))
    
    return contents

if __name__ == '__main__':
    web_path="https://www.electrolux.vn/sitemap.xml"
    filter_urls=["https://www.electrolux.vn/blog"]
    
    docs = crawler_site(web_path, filter_urls)
    print(docs)

