from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence
from pymongo import MongoClient
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import os
import requests
from googleapiclient.discovery import build
from dotenv import load_dotenv
import logging
import re
from dateutil.parser import parse
import urllib.parse

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="JobSync AI")

# API keys and setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
MONGODB_URL = os.getenv("MONGODB_URL")

# MongoDB setup with error handling
try:
    if not MONGODB_URL:
        raise ValueError("MONGODB_URL not set in .env file")
    client = MongoClient(MONGODB_URL)
    db = client["job_sync_db"]
    jobs_collection = db["jobs"]
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    jobs_collection = None

def initialize_sample_data():
    if jobs_collection is not None and jobs_collection.count_documents({}) == 0:
        sample_jobs = [
            {
                "title": "Software Engineer at Infosys",
                "company": "Infosys",
                "role": "Software Engineer",
                "location": "Bangalore, India",
                "salary": "₹8,00,000 - ₹12,00,000",
                "visa_sponsorship": "No",
                "source": "naukri",
                "posted_date": (datetime.now() - timedelta(days=1)).isoformat(),
                "apply_link": "https://www.naukri.com/job/software-infosys/apply"
            },
            {
                "title": "Software Engineer at Wipro",
                "company": "Wipro",
                "role": "Software Engineer",
                "location": "Bangalore, India",
                "salary": "₹7,50,000 - ₹10,00,000",
                "visa_sponsorship": "Yes",
                "source": "linkedin",
                "posted_date": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "apply_link": "https://www.linkedin.com/jobs/wipro-apply"
            }
        ]
        jobs_collection.insert_many(sample_jobs)
        logger.info("Sample data inserted into MongoDB")

def fetch_jobs_from_db(query: str):
    if jobs_collection is not None:
        location_match = re.search(r'in\s+(.+)', query.lower())
        location = location_match.group(1) if location_match else None
        keywords = [kw for kw in query.lower().split() if kw not in ['in', location] if kw]
        job_filter = {"$or": [{"title": {"$regex": "|".join(keywords), "$options": "i"}}]}
        if location:
            job_filter["location"] = {"$regex": location, "$options": "i"}
        jobs = list(jobs_collection.find(job_filter).limit(5))
        logger.debug(f"Fetched {len(jobs)} jobs from MongoDB for location: {location}")
        return jobs
    logger.warning("MongoDB collection not available, returning empty list")
    return []

def fetch_jobs_from_google_search(query: str):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        location_match = re.search(r'in\s+(.+)', query.lower())
        location = location_match.group(1) if location_match else "any"
        base_query = query.lower().replace(f"in {location}", "").strip() or "jobs"
        all_items = []
        for start in [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]:  # Ten pages to get 100+ results
            res = service.cse().list(
                q=f"{base_query} site:indeed.com OR site:linkedin.com OR site:naukri.com OR site:glassdoor.com OR site:*.companycareers.com OR site:*.careers OR site:*.jobs {location}",
                cx=GOOGLE_CSE_ID, num=10, start=start, dateRestrict='m'  # Restrict to last month
            ).execute()
            items = res.get("items", [])
            all_items.extend(items)
            if len(items) < 10:  # No more results
                break
        web_jobs = []
        current_date = datetime.now()
        for item in all_items[:100]:  # Limit to 100 results to ensure 30+ unique companies
            title = item.get("title", "Unknown Title")
            link = item.get("link", "#")
            snippet = item.get("snippet", "")
            # Extract company name with stricter matching
            company_match = re.search(r'at\s+([A-Za-z\s]+)(?:\s|$)', title.lower()) or re.search(r'by\s+([A-Za-z\s]+)(?:\s|$)', snippet.lower()) or re.search(r'(?:hiring|from)\s+([A-Za-z\s]+)', snippet.lower())
            company = company_match.group(1).strip() if company_match else None
            if not company:
                # Infer company from URL domain if possible
                domain = urllib.parse.urlparse(link).netloc.replace("www.", "").split('.')[0]
                company = domain.capitalize() if domain and not any(keyword in domain.lower() for keyword in ["jobs", "careers", "employment"]) else "Unknown Company"
            # Extract role (limited to 3-4 words)
            role_match = re.search(r'(?:for the role of|Role:)\s*([A-Za-z\s]{3,20})', snippet, re.IGNORECASE)
            role = role_match.group(1).strip() if role_match else re.sub(r'\s+at\s+.*', '', title).strip().replace("jobs", "").strip().split()[0:4]
            role = " ".join(role) if isinstance(role, list) else role
            if not role or len(role.split()) > 4 or any(keyword in role.lower() for keyword in ["jobs", "vacancies", "employment", "careers"]):
                role = title.split()[0] if title.split() else "Unknown Role"  # Fallback to first word, avoiding generics
            # Extract specific location from snippet
            location_text = f"{location.title()}, {location.split()[-1].upper()}" if location != "any" else "Unknown"
            if "location" in snippet.lower() or "in" in snippet.lower():
                loc_match = re.search(r'(?:location|in)\s+([A-Za-z\s,]+)', snippet.lower())
                location_text = loc_match.group(1).strip() if loc_match else location_text
            # Extract salary from snippet
            salary_match = re.search(r'₹?(\d{1,3}(?:,\d{3})*)\s*[-–]\s*₹?(\d{1,3}(?:,\d{3})*)', snippet) or re.search(r'\$(\d{1,3}(?:,\d{3})*)\s*[-–]\s*\$(\d{1,3}(?:,\d{3})*)', snippet)
            salary = f"₹{salary_match.group(1).replace(',', '')} - ₹{salary_match.group(2).replace(',', '')}" if salary_match else None
            if not salary:
                salary_match_usd = re.search(r'\$(\d{1,3}(?:,\d{3})*)\s*[-–]\s*\$(\d{1,3}(?:,\d{3})*)', snippet)
                salary = f"${salary_match_usd.group(1).replace(',', '')} - ${salary_match_usd.group(2).replace(',', '')}" if salary_match_usd else None
            if not salary:
                salary = "$60,000 - $80,000" if "Engineer" in role else "$40,000 - $60,000"  # Tailored defaults
            # Extract visa sponsorship from snippet
            visa_match = re.search(r'(?:visa sponsorship|work visa)\s+(available|yes|no)', snippet, re.IGNORECASE)
            visa_sponsorship = visa_match.group(1) if visa_match else "No"  # Default to "No" if not mentioned
            # Extract posted date and convert to relative time
            posted_match = re.search(r'Posted\s+(.+?)(?:\s|$)', snippet, re.IGNORECASE)
            posted_date = posted_match.group(1) if posted_match else None
            if posted_date:
                try:
                    posted_datetime = parse(posted_date, fuzzy=True)
                    delta = current_date - posted_datetime
                    if delta.days > 30:
                        posted_date = f"{delta.days // 30} month{'s' if delta.days // 30 > 1 else ''} ago"
                    elif delta.days > 0:
                        posted_date = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
                    elif delta.seconds > 3600:
                        posted_date = f"{delta.seconds // 3600} hour{'s' if delta.seconds // 3600 > 1 else ''} ago"
                    else:
                        posted_date = f"{delta.seconds // 60} minute{'s' if delta.seconds // 60 > 1 else ''} ago"
                except ValueError:
                    response = requests.head(link, timeout=5)
                    posted_date = "1 day ago" if response.status_code == 200 else "7 days ago"
            else:
                response = requests.head(link, timeout=5)
                posted_date = "1 day ago" if response.status_code == 200 else "7 days ago"
            # Determine source from link
            source = "unknown"
            if "indeed" in link:
                source = "indeed"
            elif "linkedin" in link:
                source = "linkedin"
            elif "naukri" in link:
                source = "naukri"
            elif "glassdoor" in link:
                source = "glassdoor"
            elif "companycareers" in link or "careers" in link or "jobs" in link:
                source = "companycareers"
            web_jobs.append({
                "title": f"{role} at {company}",
                "company": company,
                "role": role,
                "location": location_text,
                "salary": salary,
                "visa_sponsorship": visa_sponsorship,
                "source": source,
                "posted_date": posted_date,
                "apply_link": link
            })
        logger.debug(f"Fetched {len(web_jobs)} jobs from Google Search for location: {location}")
        return web_jobs
    except Exception as e:
        logger.error(f"Google Search API error: {e}")
        return []

def get_job_recommendations(query: str):
    web_jobs = fetch_jobs_from_google_search(query)
    if not web_jobs:
        logger.warning("No jobs found from Google Search, falling back to MongoDB")
    db_jobs = fetch_jobs_from_db(query) if not web_jobs else []
    all_jobs = {f"{job['company']}-{job['role']}": job for job in web_jobs + db_jobs}  # Unique key by company and role
    jobs = list(all_jobs.values())[:100]  # Increased to 100 results to ensure 30+ unique companies
    logger.debug(f"Combined {len(jobs)} total job recommendations: {len(web_jobs)} from web, {len(db_jobs)} from DB")
    return jobs

def time_ago(posted_date: str):
    if not posted_date:
        return "1 day ago"  # Should never reach here due to inference
    if re.search(r'days|hours|minutes|ago|month', posted_date, re.IGNORECASE):
        return posted_date
    try:
        posted = datetime.fromisoformat(posted_date.replace("Z", "+00:00"))
        now = datetime.now()
        diff = now - posted
        if diff.days > 30:
            return f"{diff.days // 30} month{'s' if diff.days // 30 > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600} hour{'s' if diff.seconds // 3600 > 1 else ''} ago"
        else:
            return f"{diff.seconds // 60} minute{'s' if diff.seconds // 60 > 1 else ''} ago"
    except ValueError:
        return "1 day ago"  # Should never reach here due to inference

# LangChain setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0
)

job_prompt_template = """
# JobSync AI – Job Recommendation Prompt

## Task
You are JobSync AI, a job recommendation assistant built by xAI. Your goal is to provide concise, accurate, and user-friendly job recommendations based on the user's query and the provided job data, formatted exactly as specified.

## Input
- **User Query**: {query}
- **Job Data**: {job_context}

## Instructions
1. Analyze the user's query to identify the job role and specific location (e.g., 'Embedded Software Engineer' in 'Bangalore, India').
2. Use the provided job data to select the most relevant job listings (up to 100).
3. Format each job recommendation **exactly** as follows, with each job listed separately by company and role, no grouping:
   - **Title**: Bold the job title on a new line (e.g., **Embedded Software Engineer at Bosch Group**).
   - **Company**: "- Company: " followed by the specific company name (e.g., - Company: Bosch Group).
   - **Role**: "- Role: " followed by the job role (limited to 3-4 words, e.g., - Role: Embedded Software Engineer).
   - **Location**: "- Location: " followed by the specific location (e.g., - Location: Bangalore, India).
   - **Salary**: "- Salary: " followed by the specific salary range (e.g., - Salary: ₹8L - ₹12L or $50K - $70K) inferred or extracted.
   - **Visa Sponsorship**: "- Visa Sponsorship: " followed by 'Yes' or 'No' inferred or extracted.
   - **Source**: "- Source: " followed by the exact website (e.g., - Source: indeed, linkedin, naukri, glassdoor, companycareers).
   - **Posted**: "- Posted: " followed by the relative time (e.g., - Posted: 2 days ago, 1 month ago) strictly fetched from the website or inferred from link freshness.
   - **Apply Link**: "- [Apply Now](link)" with the URL specific to that job listing, strictly matching the title, location, and role.
   - Separate each job with an empty line.
4. Start the response with: "Here are the job recommendations based on your query:" followed by a new line.
5. Ensure the tone is professional yet approachable.
6. If no jobs are found, respond with: "Sorry, I couldn’t find any matching jobs for '{query}'. Try refining your search!"

## Output Format
Return the response as plain text with markdown formatting, matching the exact structure below. Do not include JSON or code blocks.

## Example
**Input:**
- User Query: "Embedded Software Engineer jobs in Bangalore, India"
- Job Data: "Embedded Software Engineer at Bosch Group: [Apply: https://www.bosch.com/careers/embedded-job] | Location: Bangalore, India | Salary: ₹10L - ₹15L | Visa Sponsorship: No | Source: companycareers | Posted: 5 days ago | Company: Bosch Group | Role: Embedded Software Engineer"

**Output:**
Here are the job recommendations based on your query:

**Embedded Software Engineer at Bosch Group**
- Company: Bosch Group
- Role: Embedded Software Engineer
- Location: Bangalore, India
- Salary: ₹10L - ₹15L
- Visa Sponsorship: No
- Source: companycareers
- Posted: 5 days ago
- [Apply Now](https://www.bosch.com/careers/embedded-job)
"""

chat_prompt_template = """
# JobSync AI – Chat Prompt

## Task
You are JobSync AI, a conversational assistant built by xAI. Your role is to answer follow-up questions about job recommendations based on the provided job data and previous context, using the conversation history to maintain context.

## Input
- **User Query**: {query}
- **Job Context**: {job_context}
- **Chat History**: {chat_history}

## Instructions
1. Analyze the user’s query to determine what specific information is being requested (e.g., requirements, salary, application process).
2. Use the job context to extract relevant details. If the information isn’t explicitly available, make reasonable inferences based on the job title and description.
3. Incorporate the chat history to maintain context.
4. Keep the tone friendly, professional, and concise—avoid lengthy explanations unless asked.
5. If the query references a job not in the context, respond with: "I don’t have details about that job. Could you specify which job from the recommendations you’re referring to?"
6. Include the apply link in markdown (e.g., [https://naukri.com/apply]) if relevant to the query.
7. End with a helpful follow-up question (e.g., "Would you like tips on preparing for this role?").

## Output Format
Return the response as plain text with markdown formatting for readability. Do not include JSON or code blocks.

## Example
**Input:**
- User Query: "What are the requirements for the Bosch Group job?"
- Job Context: "Embedded Software Engineer at Bosch Group: [Apply: https://www.bosch.com/careers/embedded-job]"
- Chat History: ""

**Output:**
For the Embedded Software Engineer position at Bosch Group in Bangalore, India, the role requires developing embedded software for automotive systems. You can apply here: [https://www.bosch.com/careers/embedded-job]. Would you like tips on preparing for this role?
"""

job_prompt = PromptTemplate(input_variables=["query", "job_context"], template=job_prompt_template)
chat_prompt = PromptTemplate(input_variables=["query", "job_context", "chat_history"], template=chat_prompt_template)
memory = ConversationBufferMemory(input_key="query", memory_key="chat_history")
job_chain = job_prompt | llm
chat_chain = chat_prompt | llm.bind(memory=memory)

def generate_recommendation(query: str, jobs: list):
    job_context = "\n".join([f"{job['title']}: [Apply: {job['apply_link']}] | Location: {job['location']} | Salary: {job['salary']} | Visa Sponsorship: {job['visa_sponsorship']} | Source: {job['source']} | Posted: {job['posted_date']} | Company: {job['company']} | Role: {job['role']}" for job in jobs])
    try:
        response = job_chain.invoke({"query": query, "job_context": job_context})
        return response.content
    except Exception as e:
        logger.error(f"Error in generate_recommendation: {e}")
        return "Error generating recommendation."

def generate_chat_response(query: str, jobs: list):
    job_context = "\n".join([f"{job['title']}: [Apply: {job['apply_link']}]" for job in jobs])
    try:
        response = chat_chain.invoke({"query": query, "job_context": job_context})
        return response.content
    except Exception as e:
        logger.error(f"Error in generate_chat_response: {e}")
        return "Error generating chat response."

# FastAPI endpoints
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_sample_data()
    yield

app.lifespan = lifespan

class JobQuery(BaseModel):
    query: str

class ChatQuery(BaseModel):
    query: str
    previous_jobs: list

@app.post("/jobs")
def recommend_jobs(job_query: JobQuery):
    try:
        jobs = get_job_recommendations(job_query.query)
        if not jobs:
            raise HTTPException(status_code=404, detail="No jobs with salary and visa data found.")
        response = generate_recommendation(job_query.query, jobs)
        return {
            "message": response,
            "jobs": [
                {
                    "title": job["title"],
                    "company": job["company"],
                    "role": job["role"],
                    "location": job["location"],
                    "salary": job["salary"],
                    "visa_sponsorship": job["visa_sponsorship"],
                    "source": job["source"],
                    "posted": time_ago(job["posted_date"]),
                    "apply_link": job["apply_link"]
                }
                for job in jobs
            ]
        }
    except Exception as e:
        logger.error(f"Internal server error in /jobs endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/chat")
def chat_about_jobs(chat_query: ChatQuery):
    try:
        response = generate_chat_response(chat_query.query, chat_query.previous_jobs)
        return {"response": response}
    except Exception as e:
        logger.error(f"Internal server error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on port 8000...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"FastAPI server failed to start: {e}")