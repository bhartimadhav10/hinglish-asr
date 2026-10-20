# Sample transcripts — unedited model output

Produced by this repository's code (`src/hinglish_asr/model.py`, via
`transcribe.py`) on 2026-10-21, running the published model
`hinglish-asr-dataset/whisper-base-hinglish-1105-en-mix` on **CPU**
(2 vCPU cloud VM, no GPU). Audio: the first clips streamed from the
public dataset
[hinglish-asr-dataset/hinglish-asr](https://huggingface.co/datasets/hinglish-asr-dataset/hinglish-asr)
— real-world conversational Hinglish (podcast-style speech about school,
teaching, and the IAS/UPSC exams). Output is Roman-script Hinglish,
exactly as the model produces it; nothing has been cleaned up.

| # | Audio | RTF (CPU) | Transcript |
|---|---:|---:|---|
| 1 | 25.0 s | 0.143 | school bahut acha tha usi school mein meri mat teacher thi aur maine sare bachchon ko parents dhamka diya ki meri maa hai dhyan rakhna |
| 2 | 3.8 s | 0.611 | Shrantik kai lagan nahi karta tha kai lagan |
| 3 | 23.1 s | 0.193 | to teacher ne number fill list bana di sir aur puchhe de ki ek class teacher ko de dena to mujhe bada acha ho gaya chemistry padhne se sahi hai kuch kar lete hain to maine seven ko 47 kar di ab ye karte hue fir do do sau ne dekh liya acha to unhone kaha ye hamara bhi kuch karo acha hum ias ki padhai karte hue ek baat kehte hain |
| 4 | 24.2 s | 0.188 | ki jab tak aap pad pe hain log salaam thokenge dekhna ye hai ki jab aap pad se hatte hain tab thokte hain ya nahi ab retire ho gaye uske baad bhi log samman kar rahe hain aur waise salaam thok dete hain iska matlab hai ki ye prestige nahi hai steam hai aur steam belongs to the person industries belong to the chair competition jaisa hamare desh mein hai kisi bhi field mein nahi |
| 5 | 19.8 s | 0.221 | safal hone ke liye aapko sau mein ek hona padega 99 ko fail hona hi hoga par darasal failure failure hota nahi hai bilkul hai na aap kisi ek cheez mein fail hue jaise aap keh rahe hain ki aap IAS ka exam apne diya nahi hoga usmein kitna accha hua yeh gaon connection jaise establish hota hai |
| 6 | 17.6 s | 0.231 | jo ab kahaniyon ka itna bada jo Yadav ka ID group sab chalate ho kaise banta matlab kitne creative contradictions kaise hote hain aaiye to paanch hazaar hai desh mein IPC paanch hazaar hai par UPSC ne bada ehsaan kiya ki kuch shandaar logon ko select nahi kiya |
| 7 | 24.0 s | 0.176 | koi select kar leti hai na to kai badhiya log khatam ho jate ek to chamatkar usme bada pareshan rehta hu mujhe apni badi duvidha hoti hai ki aisa kaise hai ki kuch logon ke paas nuksan tay hai jo hamare paas nahi hai ek mujhe guru mera jo mentalism sikhata hai aur tab mujhe samajh mein aaya ki |
| 8 | 18.8 s | 0.239 | yeh sara ka sara khel keval tricks raha hai accha pure tricks mere ghar mein jo bachche hain unko yeh bhai hai ki main chehra padh leta hoon aajkal woh mere saamne bada sambhal ke rehte hain ki kahiye padhna lein ab aaj ki generation phone ke maamle mein humse smart hi hain laptop ke maamle mein humse smart hi hain |

**Mean RTF ≈ 0.25 on CPU** — the model transcribes roughly 4× faster
than real time on a modest 2-vCPU machine, no GPU required.
Full machine-readable record: [`../results/local_verification.json`](../results/local_verification.json).
