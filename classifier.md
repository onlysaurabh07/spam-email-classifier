# Spam Email Classifier

This project trains a spam classifier with Logistic Regression and serves predictions with Streamlit.

## Features

- TF-IDF vectorization with unigrams + bigrams
- Logistic Regression model
- Model quality report in terminal (accuracy, confusion matrix, classification report)
- Streamlit web app with spam probability score

## Project Files

- `model.py` - trains model and saves artifacts
- `app.py` - Streamlit UI for predictions
- `requirements.txt` - Python dependencies
- `spam_model.pkl` - saved model (generated after training)
- `vectorizer.pkl` - saved vectorizer (generated after training)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Model

```bash
python model.py
```

Notes:
- If `spam.csv` exists locally, it will be used.
- If not, `model.py` auto-downloads a public SMS spam dataset and trains on it.

Optional custom dataset path:

```bash
python model.py --data path\to\your_dataset.csv
```

Supported dataset formats:
- columns `v1` and `v2` (classic spam.csv format)
- columns `label` and `message`
- TSV with two columns: `label<TAB>message`

## Run Streamlit App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, enter text, and click **Predict**.

## Resume Line

Built and deployed a Spam Email/SMS Classifier using NLP (TF-IDF) and Logistic Regression with a Streamlit interface for real-time prediction.
%PDF-1.4
%ùùùù ReportLab Generated PDF document http://www.reportlab.com
1 0 obj
<<
/F1 2 0 R /F2 3 0 R /F3 4 0 R
>>
endobj
2 0 obj
<<
/BaseFont /Helvetica /Encoding /WinAnsiEncoding /Name /F1 /Subtype /Type1 /Type /Font
>>
endobj
3 0 obj
<<
/BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding /Name /F2 /Subtype /Type1 /Type /Font
>>
endobj
4 0 obj
<<
/BaseFont /Courier /Encoding /WinAnsiEncoding /Name /F3 /Subtype /Type1 /Type /Font
>>
endobj
5 0 obj
<<
/Contents 10 0 R /MediaBox [ 0 0 612 792 ] /Parent 9 0 R /Resources <<
/Font 1 0 R /ProcSet [ /PDF /Text /ImageB /ImageC /ImageI ]
>> /Rotate 0 /Trans <<

>> 
  /Type /Page
>>
endobj
6 0 obj
<<
/Contents 11 0 R /MediaBox [ 0 0 612 792 ] /Parent 9 0 R /Resources <<
/Font 1 0 R /ProcSet [ /PDF /Text /ImageB /ImageC /ImageI ]
>> /Rotate 0 /Trans <<

>> 
  /Type /Page
>>
endobj
7 0 obj
<<
/PageMode /UseNone /Pages 9 0 R /Type /Catalog
>>
endobj
8 0 obj
<<
/Author (\(anonymous\)) /CreationDate (D:20260502163841+00'00') /Creator (\(unspecified\)) /Keywords () /ModDate (D:20260502163841+00'00') /Producer (ReportLab PDF Library - www.reportlab.com) 
  /Subject (\(unspecified\)) /Title (\(anonymous\)) /Trapped /False
>>
endobj
9 0 obj
<<
/Count 2 /Kids [ 5 0 R 6 0 R ] /Type /Pages
>>
endobj
10 0 obj
<<
/Filter [ /ASCII85Decode /FlateDecode ] /Length 1041
>>
stream
GatU1D/[lW&H9CNE=,#^d!A:[Nmo^U6UD\f=?r/&.<6*"op3)?DRY5mn!oT=9sU^r6[a%$GL&GHpjBKb`0Za;HP+$u#L5P;@+jH@@.d&$S1D*kc5t94S`aqpKCJr4d:QuKYrooXCQF-'&YdnpT:q3#7dE+3&R7eiLN!i'.=&^sJ@4J.'!@BDQ\p.ZSLk#_]d/>VE308<;979,E:1MeX*K*1=-X/BOA"E`)kG8n.ZumU.JUJ_@`-mZdBm2c.BR_LXdu@YH-InBXj_Ge@ZL9e]+<13[%%ktOR@oUei`RN:cMH:)Z/I!1eb=j\:R5R<Z7g]'7P<Q7<,2"gK:0&,]%u*!b!,3XY;j0mZ0rn+#HRC5CScTP.%5l[7]F:>^QC?X%pa7^o&X(6=p%e6Q<aE9oi3;2JQY!7rr,0[AVT-*R!1V2bDE_%Z.=TYGZsUM28*.bb%d!)jtdQd-HfkcDCso9kHe62A-t/j6^(8]HG[tBhHh@r\7#m(7#Cjd+qjn_nC>JV'U`n@\R<Ils\-BMcd&+;,>`#>cl9=kI&k,h=8d>RbFAk6Z<1iZ+_G!pioL$mVpgGh<LW4\8BoYK"lnM;Xu51KRu&H9#,Z9n+QmG6m1s)`[+c`.p7ZPleB)\<4Y+l0r\EGQDS1M3+,^0S'.DM5Dplrf*05aZB!J]rTQ1LKKR93r7HqC@dG)3:;*mDA.eB$@Eiun!%(ep_>EE-T%qOoL2JDk9;;qIp(AI.CLVrrn[N'Ho/D$C/CY2JDN'R/^;e/NDJr1[`(Rsi'MJ)0_>"0A[9&%2k`QfpcXQc0#:uH;7+6GI\$;ZinK(@4f$nYQrAi6G+D/7$kMY)EA0pM[7F8"\qn,q/FQT;JqNUDI^m0hu(JES:o@C!<aN]%)%C<,Vg9<.;gWY1Uok'WTC!`Ua\J5cP9OcVHo/!!q*-[*XhV:DWHcW<*28HjTLqesHm4<(OoO4baO1jP%:dd6-A`T(bSduX<f0!:hlUA^fr'1&(d5#U2/Aj1[S2=?TJ+I'INU5KgPC&HP=_)[G_V&:*Mn4Q~>endstream
endobj
11 0 obj
<<
/Filter [ /ASCII85Decode /FlateDecode ] /Length 605
>>
stream
Gat$u?#Q2d'Rf.G>Xb)Lp;$H<K-p8bS[,/0b?Eap'fg4H\MMmGc*Zojmi"`d:m@q%Gd,N7p!J<r6Frr[?.K!73ILZA5nl]#K.%1,aB,Ui]dGHOOGaN"ZC&GaLUG-me2=QJX#8AM$Tj=(HR*[J-^UVu1,T@cO?),JaIK?XCF-BB8%B@^4,<_GnrM]M^QoTA\/MdW+^%h?1S\'cl*^Zd9?`:hJ`HdMdRqdaU4,IUh&;D:U9T:2<ZhG380+!RirZq#AdmR:eAio<6+u6:C+Yi5,6"hdVNDI9fi[h5ff?0qIJD)B[]QsC+ZLcLn^=*p^W6L_1b<>_NGLIfN!m2cmpH#8Ni$hR7<6<cU?T&n(J6Es%j+Q>1e<\AK':_=>r3d@/[^LHIJn"@nj-]'P%<:P6t;&HrC7>B.b3_FW'CaB[Pt7$\,%paUC`.i`Dk$`S:-01gBOgG3q4-S`po9tPQ^HUZ"r_pMiFP?e:LoO1^d53:sWd7*_*uq39amSJ*6@LZo,^X2g?s,oHK]H:&j5lpTF-Y#WpYJF1l:uYeXZ;PC_<SAe-E&kJm'*DK2kZ)dH9#9a)7]+eP`b4m_Tu?>t)ej<Z\QrW,nu4iI~>endstream
endobj
xref
0 12
0000000000 65535 f 
0000000073 00000 n 
0000000124 00000 n 
0000000231 00000 n 
0000000343 00000 n 
0000000448 00000 n 
0000000642 00000 n 
0000000836 00000 n 
0000000904 00000 n 
0000001187 00000 n 
0000001252 00000 n 
0000002385 00000 n 
trailer
<<
/ID 
[<435816bbf85b5e96e87e99c46543c2e6><435816bbf85b5e96e87e99c46543c2e6>]
% ReportLab generated PDF document -- digest (http://www.reportlab.com)

/Info 8 0 R
/Root 7 0 R
/Size 12
>>
startxref
3081
%%EOF
