// Knowledge-graph import script (Neo4j).
// Put symptom_cleaned.csv, examination_events_cleaned.csv and examination_need_cleaned.csv
// in the Neo4j import folder (or replace the file:/// URLs with your own locations).

// Base nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Time) REQUIRE t.value IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (et:ExaminationType) REQUIRE et.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (er:ExaminationResult) REQUIRE er.value IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (cs:ConfirmedStatus) REQUIRE cs.value IS UNIQUE;

// Event uniqueness
CREATE CONSTRAINT IF NOT EXISTS
FOR (se:SymptomEvent)
REQUIRE (se.cluster_name, se.conversation_id, se.symptom) IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
FOR (ee:ExaminationEvent)
REQUIRE (ee.cluster_name, ee.conversation_id, ee.type, ee.result) IS UNIQUE;





LOAD CSV WITH HEADERS FROM 
'file:///symptom_cleaned.csv' 
AS row

WITH row
WHERE row.cluster_name IS NOT NULL 
  AND row.conversation_id IS NOT NULL

MERGE (d:Disease {name: row.cluster_name})

MERGE (c:Conversation {id: toInteger(row.conversation_id)})

MERGE (se:SymptomEvent {
    cluster_name: row.cluster_name,
    conversation_id: toInteger(row.conversation_id),
    symptom: row.symptom
})

MERGE (s:Symptom {name: row.symptom})

FOREACH (_ IN CASE WHEN row.time_cleaned IS NOT NULL AND row.time_cleaned <> "" THEN [1] ELSE [] END |
    MERGE (t:Time {value: row.time_cleaned})
    MERGE (se)-[:HAS_TIME]->(t)
)

MERGE (d)-[:HAS_SYMPTOM_EVENT]->(se)
MERGE (se)-[:HAS_SYMPTOM]->(s)
MERGE (se)-[:BELONGS_TO]->(c)

FOREACH (_ IN CASE WHEN row.confirmed IS NOT NULL THEN [1] ELSE [] END |
    MERGE (cs:ConfirmedStatus {value: row.confirmed})
    MERGE (c)-[:IN_STATUS]->(cs)
);











LOAD CSV WITH HEADERS FROM 
'file:///examination_events_cleaned.csv' 
AS row

WITH row
WHERE row.cluster_name IS NOT NULL 
  AND row.conversation_id IS NOT NULL
  AND row.examination_type IS NOT NULL 
  AND row.examination_type <> ""
  AND row.examination_result IS NOT NULL 
  AND row.examination_result <> ""

MERGE (d:Disease {name: row.cluster_name})

MERGE (c:Conversation {id: toInteger(row.conversation_id)})

MERGE (ee:ExaminationEvent {
    cluster_name: row.cluster_name,
    conversation_id: toInteger(row.conversation_id),
    type: row.examination_type,
    result: row.examination_result
})

MERGE (et:ExaminationType {name: row.examination_type})
MERGE (er:ExaminationResult {value: row.examination_result})

MERGE (d)-[:HAS_EXAM_EVENT]->(ee)
MERGE (ee)-[:HAS_TYPE]->(et)
MERGE (ee)-[:HAS_RESULT]->(er)
MERGE (ee)-[:BELONGS_TO]->(c)

FOREACH (_ IN CASE WHEN row.confirmed IS NOT NULL THEN [1] ELSE [] END |
    MERGE (cs:ConfirmedStatus {value: row.confirmed})
    MERGE (c)-[:IN_STATUS]->(cs)
);













LOAD CSV WITH HEADERS FROM 
'file:///examination_need_cleaned.csv' 
AS row

WITH row
WHERE row.conversation_id IS NOT NULL

MERGE (c:Conversation {id: toInteger(row.conversation_id)})

MERGE (et:ExaminationType {name: row.test_need})

MERGE (c)-[:HAS_FURTHER_EXAM_EVENT]->(et)

FOREACH (_ IN CASE WHEN row.confirmed IS NOT NULL THEN [1] ELSE [] END |
    MERGE (cs:ConfirmedStatus {value: row.confirmed})
    MERGE (c)-[:IN_STATUS]->(cs)
);
