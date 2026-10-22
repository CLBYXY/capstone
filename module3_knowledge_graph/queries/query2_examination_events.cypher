//////////////////////////////////////////////////
// diagnosed
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_EXAM_EVENT]->(ee:ExaminationEvent)
MATCH (ee)-[r2:HAS_TYPE]->(et:ExaminationType)
MATCH (ee)-[r3:HAS_RESULT]->(er:ExaminationResult)
MATCH (ee)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "diagnosed at the time of the visit"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, ee, et, er, c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 5

UNION

//////////////////////////////////////////////////
// no disease
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_EXAM_EVENT]->(ee:ExaminationEvent)
MATCH (ee)-[r2:HAS_TYPE]->(et:ExaminationType)
MATCH (ee)-[r3:HAS_RESULT]->(er:ExaminationResult)
MATCH (ee)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "no disease"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, ee, et, er, c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 5

UNION

//////////////////////////////////////////////////
// further examination required
//////////////////////////////////////////////////
MATCH (d:Disease)-[r1:HAS_EXAM_EVENT]->(ee:ExaminationEvent)
MATCH (ee)-[r2:HAS_TYPE]->(et:ExaminationType)
MATCH (ee)-[r3:HAS_RESULT]->(er:ExaminationResult)
MATCH (ee)-[r4:BELONGS_TO]->(c:Conversation)
MATCH (c)-[r5:IN_STATUS]->(cs:ConfirmedStatus {value: "further examination required"})
OPTIONAL MATCH (c)-[r6:HAS_FURTHER_EXAM_EVENT]->(need:ExaminationType)

RETURN d, ee, et, er, c, cs, need,
       r1, r2, r3, r4, r5, r6
LIMIT 5;