# Knowledge-graph schema

```
Disease          {name}
Conversation     {id}
Symptom          {name}
SymptomEvent     {cluster_name, conversation_id, symptom}
Time             {value}        <- string, e.g. "2 month", "ongoing"
ExaminationType  {name}
ExaminationResult {value}
ExaminationEvent {cluster_name, conversation_id, type, result}
ConfirmedStatus  {value}        <- "diagnosed at the time of the visit", "no disease", "further examination required"



Disease          -[:HAS_SYMPTOM_EVENT]->  SymptomEvent
Disease          -[:HAS_EXAM_EVENT]->     ExaminationEvent

SymptomEvent     -[:HAS_SYMPTOM]->        Symptom
SymptomEvent     -[:HAS_TIME]->           Time
SymptomEvent     -[:BELONGS_TO]->         Conversation

ExaminationEvent -[:HAS_TYPE]->           ExaminationType
ExaminationEvent -[:HAS_RESULT]->         ExaminationResult
ExaminationEvent -[:BELONGS_TO]->         Conversation

Conversation     -[:IN_STATUS]->          ConfirmedStatus
Conversation     -[:HAS_FURTHER_EXAM_EVENT]-> ExaminationType
```
