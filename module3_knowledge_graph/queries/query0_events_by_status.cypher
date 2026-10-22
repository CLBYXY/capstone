//////////////////////////////////////////////////
// diagnosed + SymptomEvent
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_SYMPTOM_EVENT]->(e)
WHERE e:SymptomEvent
MATCH (e)-[r2:HAS_SYMPTOM]->(s)
MATCH (e)-[r3:HAS_TIME]->(t)
MATCH (e)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "diagnosed at the time of the visit"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, e,
       s AS symptom, t AS time,
       NULL AS exam_type, NULL AS exam_result,
       c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 3

UNION

//////////////////////////////////////////////////
// diagnosed + ExaminationEvent
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_EXAM_EVENT]->(e)
WHERE e:ExaminationEvent
MATCH (e)-[r2:HAS_TYPE]->(et)
MATCH (e)-[r3:HAS_RESULT]->(er)
MATCH (e)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "diagnosed at the time of the visit"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, e,
       NULL AS symptom, NULL AS time,
       et AS exam_type, er AS exam_result,
       c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 3

UNION

//////////////////////////////////////////////////
// no disease + SymptomEvent
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_SYMPTOM_EVENT]->(e)
WHERE e:SymptomEvent
MATCH (e)-[r2:HAS_SYMPTOM]->(s)
MATCH (e)-[r3:HAS_TIME]->(t)
MATCH (e)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "no disease"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, e,
       s AS symptom, t AS time,
       NULL AS exam_type, NULL AS exam_result,
       c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 3

UNION

//////////////////////////////////////////////////
// no disease + ExaminationEvent
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_EXAM_EVENT]->(e)
WHERE e:ExaminationEvent
MATCH (e)-[r2:HAS_TYPE]->(et)
MATCH (e)-[r3:HAS_RESULT]->(er)
MATCH (e)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "no disease"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, e,
       NULL AS symptom, NULL AS time,
       et AS exam_type, er AS exam_result,
       c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 3

UNION

//////////////////////////////////////////////////
// further + SymptomEvent
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_SYMPTOM_EVENT]->(e)
WHERE e:SymptomEvent
MATCH (e)-[r2:HAS_SYMPTOM]->(s)
MATCH (e)-[r3:HAS_TIME]->(t)
MATCH (e)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "further examination required"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, e,
       s AS symptom, t AS time,
       NULL AS exam_type, NULL AS exam_result,
       c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 3

UNION

//////////////////////////////////////////////////
// further + ExaminationEvent
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_EXAM_EVENT]->(e)
WHERE e:ExaminationEvent
MATCH (e)-[r2:HAS_TYPE]->(et)
MATCH (e)-[r3:HAS_RESULT]->(er)
MATCH (e)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "further examination required"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, e,
       NULL AS symptom, NULL AS time,
       et AS exam_type, er AS exam_result,
       c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 3;