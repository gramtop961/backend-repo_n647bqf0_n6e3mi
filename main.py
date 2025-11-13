import os
from datetime import datetime, timezone
from typing import List, Literal, Optional, Union, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents
from bson.objectid import ObjectId

app = FastAPI(title="ERP Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models (Pydantic)
# -----------------------------
Status = Literal["running", "queued", "complete", "error"]
StepStatus = Literal["complete", "running", "queued", "error"]

class Step(BaseModel):
    name: str
    status: StepStatus = "queued"
    llm: Optional[str] = None
    progress: Optional[int] = Field(default=None, ge=0, le=100)
    duration: Optional[str] = None

class BlockMetadata(BaseModel):
    level: Optional[int] = Field(default=None, ge=1, le=3)
    formatting: Optional[List[str]] = None
    language: Optional[str] = None
    variant: Optional[str] = None
    chartType: Optional[str] = None

class Block(BaseModel):
    id: str
    type: Literal[
        "heading", "text", "table", "chart", "code", "callout", "image", "divider", "quote", "list", "embed"
    ]
    content: Any
    metadata: Optional[BlockMetadata] = None

class Task(BaseModel):
    id: Optional[str] = None
    name: str
    status: Status = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    user: str = "You"
    llm: str = "Claude Sonnet 4.5"
    startTime: Optional[str] = None
    duration: Optional[str] = None
    steps: List[Step] = Field(default_factory=list)
    canvas: List[Block] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)

class CreateTaskRequest(BaseModel):
    name: str
    user: str = "You"
    llm: str = "Claude Sonnet 4.5"
    steps: Optional[List[Step]] = None

class UpdateTaskRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[Status] = None
    progress: Optional[int] = Field(default=None, ge=0, le=100)
    user: Optional[str] = None
    llm: Optional[str] = None
    duration: Optional[str] = None
    steps: Optional[List[Step]] = None
    canvas: Optional[List[Block]] = None
    logs: Optional[List[str]] = None
    append_log: Optional[str] = None

class ChatMessage(BaseModel):
    id: Optional[str] = None
    role: Literal["user", "assistant"]
    content: str
    quickActions: Optional[List[Dict[str, Any]]] = None
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    history: List[ChatMessage]

class ChatResponse(BaseModel):
    message: ChatMessage
    readyForTask: bool = False

# -----------------------------
# Helpers
# -----------------------------

def oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def serialize_task(doc: Dict[str, Any]) -> Task:
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    # Convert datetime to iso strings
    if doc.get("startTime") and isinstance(doc["startTime"], datetime):
        doc["startTime"] = doc["startTime"].astimezone(timezone.utc).isoformat()
    return Task(**doc)

# -----------------------------
# Basic routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "ERP Orchestrator API running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but error: {str(e)[:60]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:60]}"
    return response

# -----------------------------
# Tasks
# -----------------------------
TASK_COLLECTION = "task"

@app.get("/api/tasks", response_model=List[Task])
def list_tasks(limit: int = 50):
    docs = get_documents(TASK_COLLECTION, {}, limit)
    return [serialize_task(d) for d in docs]

@app.post("/api/tasks", response_model=Task)
def create_task(req: CreateTaskRequest):
    now = datetime.now(timezone.utc).isoformat()
    task = Task(
        name=req.name,
        user=req.user,
        llm=req.llm,
        status="queued",
        progress=0,
        startTime=now,
        steps=req.steps or [
            Step(name="Analyze Data", status="queued"),
            Step(name="Write Summary", status="queued"),
            Step(name="Gen Charts", status="queued"),
            Step(name="Final Assembly", status="queued"),
        ],
        canvas=[],
        logs=["Task created"],
    )
    new_id = create_document(TASK_COLLECTION, task)
    doc = db[TASK_COLLECTION].find_one({"_id": ObjectId(new_id)})
    return serialize_task(doc)

@app.get("/api/tasks/{task_id}", response_model=Task)
def get_task(task_id: str):
    doc = db[TASK_COLLECTION].find_one({"_id": oid(task_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Task not found")
    return serialize_task(doc)

@app.patch("/api/tasks/{task_id}", response_model=Task)
def update_task(task_id: str, req: UpdateTaskRequest):
    updates: Dict[str, Any] = {k: v for k, v in req.model_dump(exclude_unset=True).items() if k != "append_log"}
    if req.append_log:
        db[TASK_COLLECTION].update_one({"_id": oid(task_id)}, {"$push": {"logs": req.append_log}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    if updates:
        updates["updated_at"] = datetime.now(timezone.utc)
        db[TASK_COLLECTION].update_one({"_id": oid(task_id)}, {"$set": updates})
    doc = db[TASK_COLLECTION].find_one({"_id": oid(task_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Task not found")
    return serialize_task(doc)

# -----------------------------
# Chat (simulated assistant)
# -----------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    last = req.history[-1] if req.history else None
    ready = False
    content = "Let's define what you want to create. Pick metrics and time period, then hit Create Task."
    quick = [
        {"label": "All metrics", "value": {"metrics": "all"}, "selected": True},
        {"label": "Q4 2024", "value": {"period": "Q4 2024"}, "selected": True},
        {"label": "Add charts", "value": {"charts": True}, "selected": False},
    ]
    if last and last.role == "user":
        txt = last.content.lower()
        if "ready" in txt or "create" in txt:
            content = "Spec looks good. Click Create Task to start the pipeline."
            ready = True
        elif "metrics" in txt or "period" in txt:
            content = "Great, noted. Anything else to include? You can add competitor analysis or forecast."
        else:
            content = f"Got it: {last.content}. Which metrics and time period should we use?"
    msg = ChatMessage(role="assistant", content=content, quickActions=quick, timestamp=datetime.now(timezone.utc).isoformat())
    return ChatResponse(message=msg, readyForTask=ready)

# -----------------------------
# Schemas endpoint (for tooling)
# -----------------------------
@app.get("/schema")
def schema_info():
    return {
        "collections": [
            {
                "name": "task",
                "fields": Task.model_json_schema(),
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
