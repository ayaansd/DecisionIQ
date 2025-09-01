
#order to execute in 
STAGES = [
    "start",
    "eda",
    "kpi_extraction",
    "charting",
    "summary",
    "qna",
    "done"
]
#define how agent transitions between STAGES
STAGE_TRANSITIONS ={
    "start":"eda",
    "eda": "kpi_extraction",
    "kpi_extraction": "charting",
    "charting": "summary",
    "summary": "qna",
    "qna": "done"
}

def initialize_state(goal_prompt = 'auto'):
    return {
        "current_stage":"start",
        "goal":goal_prompt,
        "feedback_log":[],
        "completed_stages":[]
    }
