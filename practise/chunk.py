from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Swarnapravo Tripathi
B.Tech Undergraduate, Computer Science & EngineeringEmail:swarnatripathi03@gmail.com
Heritage Institute of Technology, Kolkata
Academic Qualifications
Year Degree/Certificate Institute CGPA/%
2023 - Present B.Tech in CSE Heritage Institute of Technology, Kolkata 8.02/10
2021 CBSE(XII) D.A.V. Public School, Haldia 93.8%
2019 CBSE(X) St. Xavier’s School, Haldia 94.6%
Achievements
•Selected among theTop 50 teamsin college-level screening forSmart India Hackathon (SIH).
•Maintained aCGPA of 8.02in Computer Science & Engineering.
•Secured93.8%in CBSE Class XII and94.6%in CBSE Class X.
Key Projects
•Smart Healthcare Monitoring System (Ongoing)
Tech Stack:Python, JavaScript, MongoDB, APIs, Power BI
–Developed a healthcare monitoring platform with backend workflows, API integration, and database management.
–Performed data validation and analysis using Python and structured datasets.
–Built dashboards to visualize healthcare metrics and monitoring data.
–Collaborated to improve scalability and reliability of the system.
– GitHub:github.com/SwarnaTripathi/smart-healthcare-monitoring
•ChronoSync – Workflow & Task Monitoring Application(Hackathon Project - 2025)
Tech Stack:JavaScript, React.js, Node.js, SQL, REST APIs
–Developed backend workflows and integrated APIs for application communication.
–Managed database operations and improved workflow reliability.
–Collaborated with teammates using Git and GitHub during a hackathon development cycle.
– GitHub:github.com/lokeshbhai007/hh310
Technical Skills
•Languages:Java, Python, JavaScript, SQL
•Web Technologies:HTML5, CSS3, React.js, Node.js, REST APIs
•Databases:MySQL, MongoDB
•Data & Visualization:Power BI, MS Excel, EDA
•Concepts:OOP, DSA, DBMS, OS, API Integration
•Tools:Git, GitHub, VS Code
Positions of Responsibility
•Fest Volunteer & Organising Team Member
–Assisted in planning and execution of club and college-level events.
–Collaborated with teams to manage logistics and event operations.
–Worked in fast-paced team environments during college events.
Extra-Curricular Activities
•Active member of Drama and Literature Club; won multiple inter-college drama competitions.
•Participated in hackathons and collaborative technical events.
•Featured as a voice actor on Mirchi 98.3 Red FM with recorded audio selected for future broadcasting.
i
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
