import asyncio
import os
import httpx
import re
from typing import Any, Literal

from rank_bm25 import BM25Okapi
from src.retrieval.router_policy import ROUTER_SYSTEM_PROMPT, classify_query, route_query

QueryRoute = Literal["FACTUAL", "RELATIONAL", "COMPLEX", "GENERAL"]


def _build_bm25_index():
    docs = [
        # 0 - Malaria
        (
            "Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through "
            "infected Anopheles mosquito bites. Symptoms include high fever with chills, sweating, headache, "
            "nausea, vomiting, and muscle pain. Treatment uses artemisinin-based combination therapy (ACT) "
            "such as artemether-lumefantrine; chloroquine is used where parasites remain sensitive. "
            "Prevention includes insecticide-treated mosquito nets, DEET repellent, and prophylactic "
            "drugs like doxycycline or mefloquine for travellers."
        ),
        # 1 - Diabetes
        (
            "Diabetes mellitus is a chronic condition of impaired blood glucose regulation due to insulin "
            "deficiency or insulin resistance. Type 2 diabetes is managed with metformin, lifestyle changes, "
            "weight loss, diet, and exercise; insulin therapy is added when oral agents fail. HbA1c reflects "
            "average blood glucose over three months and guides treatment targets. Long-term complications "
            "include diabetic nephropathy, neuropathy, retinopathy, and cardiovascular disease."
        ),
        # 2 - Tuberculosis
        (
            "Tuberculosis (TB) is a bacterial infection caused by Mycobacterium tuberculosis, primarily "
            "affecting the lungs. Symptoms include a persistent cough (often with blood-stained sputum), "
            "fever, night sweats, and significant weight loss. Standard treatment is the DOTS regimen: "
            "isoniazid, rifampicin, pyrazinamide, and ethambutol for six months. Diagnosis relies on "
            "sputum smear microscopy, culture, chest X-ray, and the Mantoux tuberculin skin test."
        ),
        # 3 - Hypertension
        (
            "Hypertension (high blood pressure) is defined as sustained systolic BP above 130 mmHg or "
            "diastolic above 80 mmHg. It often causes no symptoms but can lead to headache and dizziness "
            "at very high levels. First-line treatments include ACE inhibitors, angiotensin receptor blockers, "
            "beta-blockers, thiazide diuretics, and calcium channel blockers. Lifestyle modifications — "
            "sodium restriction, regular exercise, weight loss, and limiting alcohol — are essential. "
            "Uncontrolled hypertension causes stroke, heart failure, and chronic kidney disease."
        ),
        # 4 - Asthma
        (
            "Asthma is a chronic inflammatory airway disease characterised by episodic wheezing, shortness "
            "of breath, chest tightness, and cough triggered by allergens, exercise, or cold air. "
            "Short-acting bronchodilators (salbutamol inhaler) provide acute relief; inhaled corticosteroids "
            "such as beclomethasone are the mainstay of long-term control. Severe asthma may require "
            "long-acting beta-agonists (LABA) or oral steroids. Diagnosis involves spirometry, peak flow "
            "measurement, and reversibility testing."
        ),
        # 5 - HIV / AIDS
        (
            "HIV (Human Immunodeficiency Virus) is a retrovirus that attacks CD4 T-cells, leading to "
            "AIDS if untreated. It is transmitted through unprotected sex, shared needles, and mother-to-child "
            "routes. Antiretroviral therapy (ART) — typically tenofovir, lamivudine, and dolutegravir — "
            "suppresses viral load, allowing near-normal lifespan. Monitoring involves regular CD4 count and "
            "viral load testing. Untreated HIV leads to opportunistic infections like TB and PCP."
        ),
        # 6 - Dengue
        (
            "Dengue fever is a viral illness transmitted by Aedes mosquitoes, causing sudden high fever, "
            "severe headache, retro-orbital pain, myalgia, skin rash, and thrombocytopenia (low platelet "
            "count). Severe dengue (haemorrhagic dengue) can cause bleeding and shock. Treatment is "
            "supportive: oral rehydration, paracetamol for fever, and close platelet monitoring. NSAIDs and "
            "aspirin must be avoided due to bleeding risk. Diagnosis uses NS1 antigen, IgM/IgG serology, or PCR."
        ),
        # 7 - Pneumonia
        (
            "Pneumonia is an acute lower respiratory infection causing inflammation and consolidation of "
            "lung tissue. Common symptoms include fever, productive cough, chest pain, dyspnoea, and reduced "
            "oxygen saturation. Community-acquired pneumonia is treated with amoxicillin or azithromycin; "
            "hospital-acquired or severe cases may require IV ceftriaxone or broad-spectrum antibiotics. "
            "Diagnosis involves chest X-ray, sputum culture, blood culture, and pulse oximetry."
        ),
        # 8 - Cholera
        (
            "Cholera is an acute diarrhoeal illness caused by Vibrio cholerae, characterised by profuse "
            "rice-water stools, vomiting, and rapid dehydration. Severe dehydration can be fatal within "
            "hours if untreated. The primary treatment is aggressive oral rehydration with ORS; IV Ringer's "
            "lactate is used in severe cases. Antibiotics (doxycycline or tetracycline) shorten illness "
            "duration and reduce transmission. Diagnosis is by stool culture or rapid diagnostic test."
        ),
        # 9 - Anaemia
        (
            "Anaemia is defined by haemoglobin below 13 g/dL in men or 12 g/dL in women, most commonly "
            "due to iron deficiency. Symptoms include fatigue, pallor, weakness, breathlessness on exertion, "
            "and pale conjunctivae. Iron-deficiency anaemia is treated with ferrous sulfate supplements; "
            "B12-deficiency anaemia with cyanocobalamin injections; folate-deficiency with folic acid. "
            "Diagnosis uses full blood count (MCV, MCHC), serum ferritin, and B12 levels."
        ),
        # 10 - Stroke
        (
            "A stroke occurs when cerebral blood flow is interrupted — ischaemic (clot) or haemorrhagic "
            "(bleed). Acute symptoms include sudden facial drooping, arm weakness, speech difficulty (FAST "
            "acronym), and vision changes. Ischaemic stroke is treated with IV tPA (thrombolysis) within "
            "4.5 hours; aspirin and anticoagulants prevent recurrence. Rehabilitation with physiotherapy "
            "and speech therapy addresses long-term deficits. Diagnosis requires urgent CT or MRI brain."
        ),
        # 11 - Heart Failure
        (
            "Heart failure occurs when the heart cannot pump sufficient blood to meet the body's demands, "
            "commonly due to reduced ejection fraction. Symptoms include dyspnoea, orthopnoea, paroxysmal "
            "nocturnal dyspnoea, peripheral oedema, and fatigue. Treatment combines ACE inhibitors "
            "(enalapril), beta-blockers (carvedilol), loop diuretics (furosemide), and spironolactone. "
            "BNP levels and echocardiogram assess severity. Underlying causes include coronary artery "
            "disease, hypertension, and valvular disease."
        ),
        # 12 - Hepatitis
        (
            "Viral hepatitis inflames the liver and is classified as hepatitis A, B, C, D, or E. "
            "Symptoms include jaundice, fatigue, nausea, abdominal pain, and elevated liver enzymes "
            "(ALT, AST). Hepatitis B is treated with tenofovir or entecavir; hepatitis C with direct-acting "
            "antivirals achieving >95% cure rates. Chronic hepatitis B and C can progress to cirrhosis and "
            "hepatocellular carcinoma. Diagnosis includes HBsAg, HCV antibody, liver function tests, and "
            "ultrasound or FibroScan."
        ),
        # 13 - COVID-19
        (
            "COVID-19 is caused by the SARS-CoV-2 coronavirus and spreads via respiratory droplets. "
            "Symptoms range from mild fever, dry cough, fatigue, and loss of smell/taste to severe "
            "dyspnoea, hypoxia, and respiratory failure. Treatment for severe disease includes "
            "dexamethasone, remdesivir, oxygen therapy, and ICU ventilation when needed. "
            "Vaccination with mRNA or vector vaccines provides strong protection. Diagnosis is by "
            "RT-PCR or rapid antigen tests; chest CT shows characteristic ground-glass opacities."
        ),
        # 14 - Sepsis
        (
            "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to "
            "infection. Features include fever or hypothermia, tachycardia, hypotension, elevated lactate, "
            "and signs of organ failure (the qSOFA/SOFA scores). Management requires urgent IV broad-spectrum "
            "antibiotics, aggressive fluid resuscitation, and vasopressors (norepinephrine) in septic shock. "
            "Blood cultures and procalcitonin levels guide diagnosis and antibiotic de-escalation. "
            "Early recognition and treatment in ICU significantly reduces mortality."
        ),
        # 15 - Cancer
        (
            "Cancer is characterised by uncontrolled cell growth forming a malignant tumour with potential "
            "for metastasis. Common warning signs include unexplained weight loss, fatigue, a new lump, "
            "changes in bowel habits, and abnormal bleeding. Treatment modalities include surgery, "
            "chemotherapy, radiotherapy, targeted therapy, and immunotherapy, selected by staging and "
            "histopathology. Early detection through screening (mammography, colonoscopy, PSA) improves "
            "outcomes. Tumour markers (CEA, AFP, CA-125) assist in monitoring."
        ),
        # 16 - Arthritis
        (
            "Arthritis encompasses over 100 conditions causing joint inflammation. Rheumatoid arthritis "
            "is an autoimmune disease producing synovial pannus, cartilage erosion, joint swelling, morning "
            "stiffness, and deformity. Osteoarthritis involves cartilage breakdown from wear and tear. "
            "NSAIDs (ibuprofen) relieve pain; disease-modifying drugs (methotrexate, hydroxychloroquine) "
            "slow rheumatoid arthritis progression; biologics (TNF inhibitors) are used in refractory cases. "
            "Diagnosis uses X-ray, ESR, CRP, rheumatoid factor, and anti-CCP antibodies."
        ),
        # 17 - Chronic Kidney Disease
        (
            "Chronic kidney disease (CKD) is progressive loss of kidney function, measured by declining GFR "
            "and rising serum creatinine. Causes include diabetes, hypertension, and glomerulonephritis. "
            "Symptoms appear late: fatigue, oedema, uraemia, and anaemia. Management targets the underlying "
            "cause, BP control with ACE inhibitors, dietary protein restriction, and fluid management. "
            "End-stage renal disease requires haemodialysis, peritoneal dialysis, or kidney transplantation. "
            "Proteinuria (urine albumin) is an early marker of progression."
        ),
        # 18 - Liver Disease / Cirrhosis
        (
            "Liver cirrhosis is irreversible fibrosis from chronic liver damage (alcohol, hepatitis, NAFLD). "
            "Complications include jaundice, ascites, hepatic encephalopathy, portal hypertension, and "
            "oesophageal varices. Management includes treating the underlying cause, lactulose and rifaximin "
            "for encephalopathy, diuretics (spironolactone, furosemide) for ascites, and liver transplant "
            "for end-stage disease. Liver function tests (bilirubin, albumin, PT), ultrasound, and FibroScan "
            "assess disease severity."
        ),
        # 19 - General Infection
        (
            "Infections are caused by bacteria, viruses, fungi, or parasites and trigger an immune "
            "response with fever, inflammation, leukocytosis, and elevated CRP or procalcitonin. "
            "Bacterial infections are treated with antibiotics (penicillin, cephalosporins, macrolides); "
            "viral infections with antivirals (e.g. oseltamivir for influenza); fungal infections with "
            "antifungals (fluconazole, amphotericin B). Blood culture and sensitivity testing guide "
            "antibiotic selection. Supportive care (hydration, antipyretics) is important in all infections."
        ),
        # 20 - Migraine
        (
            "Migraine is a chronic neurological disorder characterised by recurrent severe headache, "
            "nausea, vomiting, and sensitivity to light (photophobia) and sound (phonophobia). "
            "Common triggers include stress, hormonal fluctuations, sleep disturbances, certain foods "
            "(cheese, alcohol, caffeine), and bright lights. Acute attacks are treated with triptans "
            "(sumatriptan), NSAIDs (ibuprofen), and antiemetics. Preventive medications include "
            "propranolol, topiramate, and amitriptyline. Avoiding known triggers is the key lifestyle "
            "measure to reduce migraine frequency and severity."
        ),
        # 21 - GERD
        (
            "GERD (gastroesophageal reflux disease) is a chronic condition where stomach acid flows "
            "back into the oesophagus, causing heartburn, acid reflux, regurgitation, and chest discomfort. "
            "Triggers include fatty foods, caffeine, alcohol, smoking, and lying down after eating. "
            "Lifestyle changes such as weight loss, elevating the head of the bed, and avoiding trigger "
            "foods are first-line management. Medications include proton pump inhibitors (omeprazole, "
            "pantoprazole), H2 blockers (ranitidine), and antacids. Complications include oesophagitis, "
            "Barrett's oesophagus, and stricture formation."
        ),
        # 22 - Cholestasis / Chronic Cholestasis
        (
            "Cholestasis is impaired bile flow from the liver, causing accumulation of bile acids and "
            "bilirubin in the blood, leading to jaundice, pruritus (itching), pale stools, and dark urine. "
            "Chronic cholestasis results from primary biliary cholangitis, primary sclerosing cholangitis, "
            "or drug-induced liver injury. Treatment targets the underlying cause: ursodeoxycholic acid for "
            "primary biliary cholangitis, cholestyramine for itching. Liver function tests (bilirubin, "
            "alkaline phosphatase, GGT) and ultrasound are used for diagnosis. Untreated chronic "
            "cholestasis leads to liver fibrosis and cirrhosis."
        ),
        # 23 - Heart Attack (Myocardial Infarction)
        (
            "A heart attack (myocardial infarction, MI) occurs when a coronary artery is blocked by "
            "a thrombus, cutting off blood supply to the heart muscle. Symptoms include severe chest pain "
            "radiating to the left arm or jaw, sweating, shortness of breath, and nausea. Emergency "
            "treatment requires immediate aspirin, anticoagulation (heparin), and primary PCI "
            "(percutaneous coronary intervention) or thrombolysis (streptokinase, tPA) to restore blood "
            "flow. Diagnosis uses ECG (ST elevation), troponin levels, and echocardiography. Risk "
            "factors include hypertension, diabetes, smoking, hypercholesterolaemia, and obesity."
        ),
        # 24 - Hypothyroidism
        (
            "Hypothyroidism is deficiency of thyroid hormone (thyroxine, T4), most commonly due to "
            "Hashimoto's thyroiditis or iodine deficiency. Symptoms include fatigue, weight gain, cold "
            "intolerance, constipation, dry skin, hair loss, bradycardia, and depression. Diagnosis is "
            "confirmed by elevated TSH and low free T4 on blood tests. Treatment is lifelong thyroid "
            "hormone replacement with levothyroxine, with dose adjusted by TSH monitoring. Untreated "
            "hypothyroidism can cause myxoedema coma, hyperlipidaemia, and cardiovascular disease."
        ),
        # 25 - Hyperthyroidism
        (
            "Hyperthyroidism is excess thyroid hormone production, most often due to Graves' disease, "
            "toxic multinodular goitre, or thyroiditis. Symptoms include weight loss, heat intolerance, "
            "palpitations, tachycardia, anxiety, tremor, and exophthalmos (in Graves' disease). "
            "Diagnosis uses low TSH with elevated free T3/T4 and thyroid antibody testing. Treatment "
            "options include antithyroid drugs (carbimazole, propylthiouracil), radioiodine ablation, "
            "or thyroidectomy. Beta-blockers (propranolol) control symptoms of tachycardia and tremor. "
            "Untreated hyperthyroidism can cause thyroid storm, atrial fibrillation, and osteoporosis."
        ),
        # 26 - Common Cold
        (
            "The common cold is a viral upper respiratory tract infection, most often caused by "
            "rhinovirus. Symptoms include nasal congestion, runny nose, sneezing, sore throat, mild "
            "fever, and cough. Treatment is symptomatic: rest, hydration, paracetamol for fever, "
            "decongestants (pseudoephedrine), and antihistamines. Antibiotics are not effective against "
            "viral infections and should not be prescribed. Complications include secondary bacterial "
            "sinusitis, otitis media, and in asthma patients, bronchospasm and exacerbation of airway "
            "inflammation with wheezing."
        ),
        # 27 - Alcoholic Hepatitis
        (
            "Alcoholic hepatitis is acute liver inflammation caused by heavy alcohol consumption. "
            "It presents with jaundice, fever, right upper quadrant pain, nausea, and elevated liver "
            "enzymes (AST, ALT) with an AST:ALT ratio >2:1. Bilirubin is elevated and the "
            "prothrombin time is prolonged. Severe cases (Maddrey discriminant function >32) are "
            "treated with corticosteroids (prednisolone) or pentoxifylline. The most important "
            "intervention is complete alcohol abstinence. Complications include liver failure, "
            "hepatic encephalopathy, portal hypertension, and progression to cirrhosis."
        ),
        # 28 - Urinary Tract Infection
        (
            "Urinary tract infections (UTIs) are caused by bacteria, most commonly Escherichia coli, "
            "entering the urinary system. Symptoms include dysuria (painful urination), urinary "
            "frequency, urgency, suprapubic pain, and cloudy or foul-smelling urine. Upper UTI "
            "(pyelonephritis) causes fever, flank pain, and rigors. Diagnosis uses urine dipstick "
            "(positive leucocytes and nitrites) and urine culture. Treatment is antibiotics: "
            "trimethoprim, nitrofurantoin, or ciprofloxacin for uncomplicated UTI; IV cephalosporins "
            "for pyelonephritis. Recurrent UTIs in diabetic patients may lead to renal complications."
        ),
        # 29 - Depression
        (
            "Depression is a common mental health disorder characterised by persistent low mood, "
            "loss of interest (anhedonia), fatigue, changes in sleep and appetite, poor concentration, "
            "and feelings of hopelessness. It is diagnosed using DSM-5 criteria when symptoms persist "
            "for at least two weeks. Treatment includes psychotherapy (cognitive behavioural therapy, "
            "CBT), antidepressants (SSRIs: fluoxetine, sertraline; SNRIs: venlafaxine), and lifestyle "
            "modifications. Severe depression may require ECT (electroconvulsive therapy). Thyroid "
            "dysfunction, hypothyroidism in particular, can cause or worsen depressive symptoms."
        ),
        # 30 - Typhoid
        (
            "Typhoid fever is a systemic bacterial infection caused by Salmonella typhi, transmitted "
            "via contaminated food and water. Symptoms include sustained high fever (step-ladder pattern), "
            "headache, abdominal pain, constipation or diarrhoea, rose spots (rash), and splenomegaly. "
            "Diagnosis uses blood culture (gold standard), Widal test, and stool culture. Treatment is "
            "antibiotics: azithromycin, fluoroquinolones (ciprofloxacin), or third-generation "
            "cephalosporins (ceftriaxone) for resistant strains. Complications include intestinal "
            "perforation, haemorrhage, and sepsis. Prevention uses typhoid vaccines and safe water."
        ),
        # 31 - Jaundice
        (
            "Jaundice is yellowing of the skin and sclerae due to elevated bilirubin levels. It is "
            "classified as pre-hepatic (haemolysis — e.g., malaria, sickle cell), hepatic (liver "
            "dysfunction — e.g., hepatitis, cirrhosis, alcoholic liver disease), or post-hepatic "
            "(bile duct obstruction — e.g., gallstones, cholangiocarcinoma). In malaria, haemolysis "
            "of red blood cells releases excess bilirubin, causing jaundice alongside fever, chills, "
            "and anaemia. Management targets the underlying cause: antimalarial treatment (chloroquine, "
            "artemisinin) for malaria-induced jaundice; antiviral or supportive care for hepatitis. "
            "Liver function tests (bilirubin, ALT, ALP) guide diagnosis."
        ),
    ]
    tokenized = [d.lower().split() for d in docs]
    print(f"[BM25] Built index with {len(docs)} documents")
    return BM25Okapi(tokenized), docs


BM25_INDEX, BM25_CORPUS = _build_bm25_index()


def _context_fallback_answer(query: str, context: str) -> str:
    cleaned = " ".join((context or "").split())
    if cleaned:
        snippet = cleaned[:420]
        return (
            f"Based on the retrieved clinical context for '{query}': {snippet}. "
            "Please consult a licensed clinician for patient-specific treatment decisions."
        )
    return (
        f"Medical information for query: {query}. Common management depends on confirmed diagnosis, "
        "clinical severity, and comorbidities."
    )


def call_llm_direct(query: str, context: str) -> str:
    model = os.getenv("LLM_MODEL", "llama3.1:8b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if context and len(context.strip()) > 30:
        prompt = (
            f"You are a helpful medical assistant. Use the context below and your knowledge "
            f"to answer the question clearly and accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer directly and specifically in 3-4 sentences:"
        )
    else:
        prompt = (
            f"You are a helpful, knowledgeable medical assistant. "
            f"Answer the following medical question clearly and accurately.\n\n"
            f"Question: {query}\n\n"
            f"Answer in 3-4 sentences:"
        )
    try:
        r = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 250, "temperature": 0.1}},
            timeout=60.0,
        )
        out = (r.json().get("response") or "").strip()
        if len(out) > 20:
            return out
        fallback = call_llm_knowledge_only(query, model)
        return fallback if len(fallback) > 20 else _context_fallback_answer(query, context)
    except Exception as e:
        print(f"LLM_ERROR: {e}")
        fallback = call_llm_knowledge_only(query, model)
        return fallback if len(fallback) > 20 else _context_fallback_answer(query, context)


def call_llm_knowledge_only(query: str, model: str) -> str:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        r = httpx.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": (
                    f"You are a helpful, knowledgeable medical assistant. "
                    f"Answer this question clearly and accurately: {query}"
                ),
                "stream": False,
                "options": {"num_predict": 250, "temperature": 0.1},
            },
            timeout=60.0,
        )
        resp = r.json().get("response", "").strip()
        return resp if len(resp) > 20 else ""
    except Exception as e:
        print(f"LLM_KNOWLEDGE_ERROR: {e}")
        return ""


def bm25_retrieve_sync(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    tokens = (query or "").lower().split()
    scores = BM25_INDEX.get_scores(tokens)

    # Sort by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        # Remove score>0 filter - always return top_k documents
        results.append({
            "content": BM25_CORPUS[i][:400],
            "metadata": {"score": float(scores[i]), "source": "bm25"}
        })

    return results


async def bm25_retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return await asyncio.to_thread(bm25_retrieve_sync, query, top_k)


def _is_empty_or_unavailable(text: str) -> bool:
    lowered = (text or "").lower()
    return (not lowered.strip()) or ("unable to process request" in lowered)


def _general_medical_answer(query: str) -> str:
    return call_llm_direct(query, "")


def _build_answer(query: str, docs: list[dict[str, Any]]) -> tuple[str, str]:
    context = "\n\n".join([d.get("content", "") for d in docs])
    answer = call_llm_direct(query, context)
    return context, answer


def hybrid_retrieve(query: str) -> dict[str, Any]:
    route = classify_query(query)
    if route == "RELATIONAL":
        docs = bm25_retrieve_sync(query, top_k=3)
    elif route == "FACTUAL":
        docs = bm25_retrieve_sync(query, top_k=5)
    elif route == "COMPLEX":
        docs = bm25_retrieve_sync(query, top_k=4)
    else:
        docs = bm25_retrieve_sync(query, top_k=3)
    context, answer = _build_answer(query, docs)
    return {"route": route, "context": context, "answer": answer, "sources": docs, "steps": [f"route={route}"], "evidence": {"bm25_hits": docs}}


def hybrid_retrieve_no_routing(query: str) -> dict[str, Any]:
    docs = bm25_retrieve_sync(query, top_k=2)
    context, answer = _build_answer(query, docs)
    return {"route": "HYBRID", "context": context, "answer": answer, "sources": docs, "steps": ["route=HYBRID"], "evidence": {"bm25_hits": docs}}
