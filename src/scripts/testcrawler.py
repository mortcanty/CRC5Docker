import sys
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

browser_config = BrowserConfig(
    headless=True,
    text_mode=True
)

async def main(ws, md):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=md,
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False
    )

    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(ws, config=config)

        print(f"Crawled {len(results)} pages in total")

        # Access individual results
        with open("/home/mort/crawl4ai.md", 'w') as file:
            for result in results:
                print(f"URL: {result.url}")
                file.write(f"{result.markdown}")
                print(f"Depth: {result.metadata.get('depth', 0)}")

# Run the async main function
asyncio.run(main(sys.argv[1], float(sys.argv[2])))
