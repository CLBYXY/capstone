//////////////////////////////////////////////////
// diagnosed
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
MATCH (se)-[r2:HAS_SYMPTOM]->(s:Symptom)
MATCH (se)-[r3:HAS_TIME]->(t:Time)
MATCH (se)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "diagnosed at the time of the visit"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(et:ExaminationType)

RETURN d, se, s, t, c, cs, et,
       r1, r2, r3, r4, r5, r6
LIMIT 5

UNION

//////////////////////////////////////////////////
// no disease
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
MATCH (se)-[r2:HAS_SYMPTOM]->(s:Symptom)
MATCH (se)-[r3:HAS_TIME]->(t:Time)
MATCH (se)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "no disease"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(et:ExaminationType)

RETURN d, se, s, t, c, cs, et,
       r1, r2, r3, r4, r5, r6
LIMIT 5

UNION

//////////////////////////////////////////////////
// further exam
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_SYMPTOM_EVENT]->(se:SymptomEvent)
MATCH (se)-[r2:HAS_SYMPTOM]->(s:Symptom)
MATCH (se)-[r3:HAS_TIME]->(t:Time)
MATCH (se)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "further examination required"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(et:ExaminationType)

RETURN d, se, s, t, c, cs, et,
       r1, r2, r3, r4, r5, r6
LIMIT 5;