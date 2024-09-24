from notion_client import Client
import os
from dotenv import load_dotenv
load_dotenv()

notion = Client(auth=os.environ["NOTION_TOKEN"])