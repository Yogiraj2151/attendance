
# =========================================================
# AI SMART ATTENDANCE MANAGEMENT SYSTEM
# =========================================================
# Streamlit + Supabase + OpenCV LBPH
# Python 3.11
#
# FEATURES
# ---------------------------------------------------------
# Admin Login
# Student Login / Signup
# Student Registration
# Face Registration
# LBPH Face Recognition
# Student Face Verification
# Admin Face Attendance
# Daily Attendance
# Monthly Attendance
# Present / Absent
# Attendance Percentage
# Student Search
# CSV Export
# Marks Management
# Student Dashboard
# Admin Dashboard
# =========================================================

import os
import calendar
from pathlib import Path
from datetime import datetime, date
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from supabase import create_client


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# =========================================================
# 🚗 AI SMART ATTENDANCE — ANIMATED CAR + MOUSE PARTICLES
# =========================================================
# PREMIUM ANIMATED BACKGROUND
# Mouse Particles + Moving Car
# =========================================================

components.html("""
<!DOCTYPE html>
<html>
<head>
<style>

html, body {
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: transparent;
}

#fx {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 999999;
}

#car {
    position: fixed;
    left: 50%;
    bottom: 7%;
    transform: translateX(-50%);
    font-size: 55px;
    filter:
        drop-shadow(0 0 8px #ff4fa3)
        drop-shadow(0 0 18px #a855f7);
    transition: transform 0.12s ease-out;
    z-index: 1000000;
}

</style>
</head>

<body>

<canvas id="fx"></canvas>

<div id="car">🏎️</div>

<script>

const canvas = document.getElementById("fx");
const ctx = canvas.getContext("2d");
const car = document.getElementById("car");

let W = window.innerWidth;
let H = window.innerHeight;

canvas.width = W;
canvas.height = H;

let mouse = {
    x: W / 2,
    y: H / 2
};

let particles = [];

window.addEventListener("resize", () => {

    W = window.innerWidth;
    H = window.innerHeight;

    canvas.width = W;
    canvas.height = H;

});

window.addEventListener("mousemove", (e) => {

    mouse.x = e.clientX;
    mouse.y = e.clientY;

    // Mouse particles
    for (let i = 0; i < 5; i++) {

        particles.push({

            x: mouse.x + (Math.random() - 0.5) * 20,
            y: mouse.y + (Math.random() - 0.5) * 20,

            vx: (Math.random() - 0.5) * 2.5,
            vy: (Math.random() - 0.5) * 2.5,

            size: Math.random() * 3 + 1,

            life: 1

        });

    }

    // Car follows mouse horizontally
    const carX =
        ((mouse.x / W) - 0.5) * 500;

    const tilt =
        ((mouse.x / W) - 0.5) * 12;

    car.style.transform =
        `translateX(calc(-50% + ${carX}px))
         rotate(${tilt}deg)`;

});

function drawParticles() {

    ctx.clearRect(0, 0, W, H);

    for (let i = particles.length - 1; i >= 0; i--) {

        const p = particles[i];

        p.x += p.vx;
        p.y += p.vy;

        p.life -= 0.018;
        p.size *= 0.985;

        if (p.life <= 0) {

            particles.splice(i, 1);
            continue;

        }

        const glow =
            ctx.createRadialGradient(
                p.x,
                p.y,
                0,
                p.x,
                p.y,
                18
            );

        glow.addColorStop(
            0,
            `rgba(255, 105, 180, ${p.life})`
        );

        glow.addColorStop(
            0.45,
            `rgba(168, 85, 247, ${p.life * 0.8})`
        );

        glow.addColorStop(
            1,
            "rgba(168, 85, 247, 0)"
        );

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            p.size,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = glow;
        ctx.fill();

    }

    requestAnimationFrame(drawParticles);

}

drawParticles();

</script>

</body>
</html>
""", height=0)
def load_css():

    css_file = Path(__file__).parent / "style.css"

    if css_file.exists():

        with open(
            css_file,
            "r",
            encoding="utf-8"
        ) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

ADMIN_EMAIL = "yogirajgole9@gmail.com"


if not SUPABASE_URL or not SUPABASE_KEY:

    st.error(
        "❌ SUPABASE_URL / SUPABASE_KEY missing."
    )

    st.info(
        "Project folder मध्ये .env file तयार करा."
    )

    st.code(
        "SUPABASE_URL=your_supabase_url\n"
        "SUPABASE_KEY=your_supabase_key"
    )

    st.stop()


# =========================================================
# SUPABASE
# =========================================================

try:

    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

except Exception as e:

    st.error(
        "❌ Supabase connection failed."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {

    "logged_in": False,

    "user": None,

    "role": None,

    "page": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

KNOWN_FACES_DIR = (
    BASE_DIR / "known_faces"
)

KNOWN_FACES_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# FACE CASCADE
# =========================================================

def load_face_cascade():

    possible_paths = [

        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml",

        BASE_DIR
        / "haarcascade_frontalface_default.xml",

        BASE_DIR
        / "haarcascade"
        / "haarcascade_frontalface_default.xml",

        BASE_DIR
        / "data"
        / "haarcascade_frontalface_default.xml"
    ]

    for path in possible_paths:

        if path.exists():

            cascade = cv2.CascadeClassifier(
                str(path)
            )

            if not cascade.empty():

                return cascade

    return None


face_cascade = load_face_cascade()


if face_cascade is None:

    st.error(
        "❌ Haar Cascade load failed."
    )

    st.info(
        "haarcascade_frontalface_default.xml "
        "project folder किंवा haarcascade folder मध्ये ठेवा."
    )

    st.stop()


# =========================================================
# DATABASE HELPERS
# =========================================================

def get_students():

    try:

        response = (
            supabase
            .table("students")
            .select("*")
            .execute()
        )

        return pd.DataFrame(
            response.data or []
        )

    except Exception as e:

        st.error(
            f"❌ Students database error: {e}"
        )

        return pd.DataFrame()


def get_attendance():

    try:

        response = (
            supabase
            .table("attendance")
            .select("*")
            .execute()
        )

        return pd.DataFrame(
            response.data or []
        )

    except Exception as e:

        st.error(
            f"❌ Attendance database error: {e}"
        )

        return pd.DataFrame()


def get_marks():

    try:

        response = (
            supabase
            .table("marks")
            .select("*")
            .execute()
        )

        return pd.DataFrame(
            response.data or []
        )

    except Exception as e:

        st.error(
            f"❌ Marks database error: {e}"
        )

        return pd.DataFrame()



# =========================================================
# HOLIDAY MANAGEMENT
# =========================================================

def get_holidays():

    try:

        response = (
            supabase
            .table("holidays")
            .select("*")
            .order("holiday_date")
            .execute()
        )

        return pd.DataFrame(response.data or [])

    except Exception as e:

        st.error(f"❌ Holidays database error: {e}")
        return pd.DataFrame()


def is_holiday(check_date):

    try:

        holiday_date = (
            check_date.isoformat()
            if isinstance(check_date, (date, datetime))
            else str(check_date)[:10]
        )

        response = (
            supabase
            .table("holidays")
            .select("holiday_date,holiday_name")
            .eq("holiday_date", holiday_date)
            .limit(1)
            .execute()
        )

        return bool(response.data), (
            response.data[0].get("holiday_name", "Holiday")
            if response.data else ""
        )

    except Exception:

        return False, ""


def holiday_management():

    st.header("🏖️ Holiday Management")
    st.caption("Admin can add and remove holidays. Attendance is automatically blocked on holiday dates.")

    st.subheader("➕ Add Holiday")

    with st.form("add_holiday_form"):

        c1, c2 = st.columns(2)

        with c1:
            holiday_date = st.date_input(
                "📅 Holiday Date",
                value=date.today(),
                key="holiday_date_input"
            )

        with c2:
            holiday_name = st.text_input(
                "🎉 Holiday Name",
                placeholder="Example: Independence Day",
                key="holiday_name_input"
            )

        add = st.form_submit_button(
            "➕ Add Holiday",
            use_container_width=True
        )

    if add:

        if not holiday_name.strip():
            st.warning("⚠️ Holiday name is required.")
            return

        try:

            existing = (
                supabase
                .table("holidays")
                .select("id")
                .eq("holiday_date", holiday_date.isoformat())
                .execute()
            )

            if existing.data:
                st.warning("⚠️ A holiday already exists for this date.")
                return

            result = (
                supabase
                .table("holidays")
                .insert({
                    "holiday_date": holiday_date.isoformat(),
                    "holiday_name": holiday_name.strip()
                })
                .execute()
            )

            if result.data:
                st.success(f"✅ {holiday_name.strip()} added for {holiday_date.strftime('%d %B %Y')}.")
                st.rerun()
            else:
                st.error("❌ Holiday could not be added.")

        except Exception as e:
            st.error("❌ Holiday save failed.")
            st.caption(str(e))

    st.divider()
    st.subheader("📋 Holiday List")

    holidays = get_holidays()

    if holidays.empty:
        st.info("No holidays added yet.")
        return

    if "holiday_date" in holidays.columns:
        holidays["holiday_date"] = pd.to_datetime(
            holidays["holiday_date"], errors="coerce"
        )
        holidays = holidays.sort_values("holiday_date")

    display = holidays.copy()

    if "holiday_date" in display.columns:
        display["holiday_date"] = display["holiday_date"].dt.strftime("%d-%m-%Y")

    columns = [
        c for c in ["id", "holiday_date", "holiday_name"]
        if c in display.columns
    ]

    st.dataframe(
        display[columns],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🗑️ Remove Holiday")

    if "id" not in holidays.columns:
        st.error("❌ Holidays table must contain an id column.")
        return

    options = []
    option_map = {}

    for _, row in holidays.iterrows():
        holiday_id = row.get("id")
        holiday_dt = row.get("holiday_date")
        holiday_name = str(row.get("holiday_name", "Holiday"))

        if pd.notna(holiday_dt):
            label = f"{holiday_dt.strftime('%d-%m-%Y')} — {holiday_name}"
        else:
            label = f"{holiday_id} — {holiday_name}"

        options.append(label)
        option_map[label] = holiday_id

    selected = st.selectbox(
        "Select holiday to remove",
        options,
        key="delete_holiday_select"
    )

    if st.button(
        "🗑️ Delete Selected Holiday",
        use_container_width=True,
        key="delete_holiday_button"
    ):

        try:
            holiday_id = option_map[selected]

            result = (
                supabase
                .table("holidays")
                .delete()
                .eq("id", holiday_id)
                .execute()
            )

            if result.data is not None:
                st.success("✅ Holiday deleted successfully.")
                st.rerun()
            else:
                st.error("❌ Holiday deletion failed.")

        except Exception as e:
            st.error("❌ Holiday deletion failed.")
            st.caption(str(e))


# =========================================================
# LOGIN
# =========================================================

def login_page():

    st.markdown(
        '<div class="main-title">'
        '🤖 AI SMART ATTENDANCE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Face Recognition • Attendance • Marks Management'
        '</div>',
        unsafe_allow_html=True
    )

    _, center, _ = st.columns(
        [1, 2, 1]
    )

    with center:

        st.subheader("🔐 Login")

        email = st.text_input(
            "📧 Email",
            key="login_email"
        )

        password = st.text_input(
            "🔑 Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "🚀 Login",
            use_container_width=True,
            key="login_button"
        ):

            email = email.strip().lower()

            if not email or not password:

                st.warning(
                    "⚠️ Email आणि Password enter करा."
                )

                return

            try:

                response = (
                    supabase
                    .auth
                    .sign_in_with_password(
                        {
                            "email": email,
                            "password": password
                        }
                    )
                )

                if response.user:

                    st.session_state.logged_in = True

                    st.session_state.user = (
                        response.user
                    )

                    if (
                        email
                        == ADMIN_EMAIL.lower()
                    ):

                        st.session_state.role = (
                            "admin"
                        )

                    else:

                        st.session_state.role = (
                            "student"
                        )

                    st.success(
                        "✅ Login successful!"
                    )

                    st.rerun()

            except Exception as e:

                st.error(
                    "❌ Invalid email or password."
                )


# =========================================================
# SIGNUP
# =========================================================

def signup_page():

    st.markdown(
        '<div class="main-title">'
        '📝 CREATE ACCOUNT'
        '</div>',
        unsafe_allow_html=True
    )

    _, center, _ = st.columns(
        [1, 2, 1]
    )

    with center:

        name = st.text_input(
            "👤 Full Name",
            key="signup_name"
        )

        email = st.text_input(
            "📧 Email",
            key="signup_email"
        )

        password = st.text_input(
            "🔑 Password",
            type="password",
            key="signup_password"
        )

        confirm = st.text_input(
            "🔐 Confirm Password",
            type="password",
            key="signup_confirm"
        )

        if st.button(
            "🚀 Create Account",
            use_container_width=True,
            key="signup_button"
        ):

            if not name.strip():

                st.warning(
                    "⚠️ Name required."
                )

                return

            if not email.strip():

                st.warning(
                    "⚠️ Email required."
                )

                return

            if len(password) < 6:

                st.warning(
                    "⚠️ Password minimum 6 characters."
                )

                return

            if password != confirm:

                st.error(
                    "❌ Password does not match."
                )

                return

            if (
                email.strip().lower()
                == ADMIN_EMAIL.lower()
            ):

                st.error(
                    "❌ हा Admin email आहे."
                )

                return

            try:

                response = (
                    supabase
                    .auth
                    .sign_up(
                        {
                            "email":
                                email.strip().lower(),

                            "password":
                                password
                        }
                    )
                )

                if response.user:

                    st.success(
                        "✅ Account created successfully!"
                    )

                    st.info(
                        "Admin ने student registration पूर्ण करणे आवश्यक आहे."
                    )

            except Exception as e:

                st.error(
                    "❌ Signup failed."
                )

                st.caption(
                    str(e)
                )


# =========================================================
# LOGGED STUDENT
# =========================================================

def get_logged_student():

    user = st.session_state.user

    if user is None:

        return None

    email = str(
        user.email
    ).strip().lower()

    students = get_students()

    if students.empty:

        return None

    if "email" not in students.columns:

        return None

    rows = students[
        students["email"]
        .astype(str)
        .str.strip()
        .str.lower()
        == email
    ]

    if rows.empty:

        return None

    return rows.iloc[0]


# =========================================================
# FACE DETECTION
# =========================================================

def detect_single_face(image):

    if image is None:

        return None, None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:

        return None, None

    if len(faces) > 1:

        return "multiple", None

    x, y, w, h = faces[0]

    face = gray[
        y:y+h,
        x:x+w
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    return face, faces[0]


# =========================================================
# SAFE FACE CHECK
# =========================================================

def is_multiple_face(face):

    return (
        isinstance(face, str)
        and face == "multiple"
    )


# =========================================================
# SUPABASE FACE STORAGE HELPERS
# =========================================================

FACE_BUCKET = "student-faces"


def upload_face_image(student_id, image, roll_no):
    """Upload a full student photo to private Supabase Storage."""
    try:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return False, None, "Could not encode the face photo."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        storage_path = f"{student_id}/{roll_no}_{timestamp}.jpg"

        supabase.storage.from_(FACE_BUCKET).upload(
            path=storage_path,
            file=encoded.tobytes(),
            file_options={
                "content-type": "image/jpeg",
                "upsert": "false",
            },
        )
        return True, storage_path, None
    except Exception as exc:
        return False, None, str(exc)


def download_face_image(storage_path):
    """Download a face photo from Supabase Storage and decode it for OpenCV."""
    try:
        image_bytes = (
            supabase.storage
            .from_(FACE_BUCKET)
            .download(storage_path)
        )
        if not image_bytes:
            return None
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


# =========================================================
# CREATE LBPH MODEL
# =========================================================

def create_recognizer():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        return None, (
            "❌ cv2.face available नाही. Install opencv-contrib-python-headless."
        )

    students = get_students()
    if students.empty:
        return None, "❌ No students registered."

    faces = []
    labels = []
    label_to_roll = {}
    label_id = 0

    for _, student in students.iterrows():
        roll_no = str(student.get("roll_no", "")).strip()
        if not roll_no:
            continue

        candidate_images = []

        # Prefer images saved in Supabase Storage.
        paths = student.get("face_image_paths", [])
        if isinstance(paths, str):
            try:
                import json
                paths = json.loads(paths)
            except Exception:
                paths = []
        if not isinstance(paths, list):
            paths = []

        for storage_path in paths:
            if not storage_path:
                continue
            image = download_face_image(str(storage_path))
            if image is not None:
                candidate_images.append(image)

        # Backward compatibility: use the local cache if no Storage image is available.
        if not candidate_images:
            for ext in (".jpg", ".jpeg", ".png"):
                path = KNOWN_FACES_DIR / f"{roll_no}{ext}"
                if path.exists():
                    image = cv2.imread(str(path))
                    if image is not None:
                        candidate_images.append(image)
                    break

        for image in candidate_images:
            face, _ = detect_single_face(image)
            if face is None or is_multiple_face(face):
                continue

            faces.append(face)
            labels.append(label_id)
            label_to_roll[label_id] = roll_no
            label_id += 1

    if not faces:
        return None, (
            "❌ Supabase Storage/local मध्ये registered face images सापडल्या नाहीत. "
            "Student Registration पुन्हा करून पाहा आणि student-faces bucket तपासा."
        )

    recognizer.train(faces, np.array(labels, dtype=np.int32))
    return (recognizer, label_to_roll), None


# =========================================================
# MARK ATTENDANCE
# =========================================================

def mark_attendance(roll_no):

    try:

        response = (
            supabase
            .table("students")
            .select("*")
            .eq(
                "roll_no",
                str(roll_no)
            )
            .execute()
        )

        if not response.data:

            return False, (
                "❌ Student not found."
            )

        student = response.data[0]

        today = date.today().isoformat()

        holiday, holiday_name = is_holiday(date.today())

        if holiday:
            return False, (
                f"🏖️ Today is a holiday: {holiday_name}. Attendance is not required."
            )

        existing = (
            supabase
            .table("attendance")
            .select("*")
            .eq(
                "roll_no",
                str(roll_no)
            )
            .eq(
                "date",
                today
            )
            .execute()
        )

        if existing.data:

            return False, (
                f"⚠️ {student['name']} "
                "ची आजची attendance already marked आहे."
            )

        now = datetime.now()

        data = {

            "roll_no":
                str(student["roll_no"]),

            "name":
                str(student["name"]),

            "department":
                str(
                    student.get(
                        "department",
                        ""
                    )
                ),

            "date":
                today,

            "time":
                now.strftime("%H:%M:%S"),

            "status":
                "Present"
        }

        result = (
            supabase
            .table("attendance")
            .insert(data)
            .execute()
        )

        if result.data:

            return True, (
                f"✅ Attendance marked for "
                f"{student['name']}"
            )

        return False, (
            "❌ Attendance insert failed."
        )

    except Exception as e:

        return False, str(e)


# =========================================================
# STUDENT FACE ATTENDANCE
# =========================================================

def student_face_attendance():

    st.header(
        "📷 Mark My Attendance"
    )

    student = get_logged_student()

    if student is None:

        st.error(
            "❌ Student registration record सापडला नाही."
        )

        return

    roll_no = str(
        student["roll_no"]
    ).strip()

    st.info(
        f"👨‍🎓 **{student['name']}** | "
        f"Roll No: **{roll_no}**"
    )

    today = date.today().isoformat()

    try:

        existing = (
            supabase
            .table("attendance")
            .select("*")
            .eq(
                "roll_no",
                roll_no
            )
            .eq(
                "date",
                today
            )
            .execute()
        )

        if existing.data:

            st.success(
                "✅ आजची attendance already marked आहे."
            )

            st.dataframe(
                pd.DataFrame(
                    existing.data
                ),
                use_container_width=True,
                hide_index=True
            )

            return

    except Exception as e:

        st.error(str(e))

        return

    result, error = create_recognizer()

    if error:

        st.error(error)

        return

    recognizer, label_to_roll = result

    camera = st.camera_input(
        "📷 Capture Your Face",
        key="student_camera"
    )

    if camera is None:

        st.info(
            "📷 Clear face photo capture करा."
        )

        return

    image_array = np.frombuffer(
        camera.getvalue(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "❌ Camera image read failed."
        )

        return

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:

        st.error(
            "❌ Face detected नाही."
        )

        return

    if len(faces) > 1:

        st.warning(
            "⚠️ फक्त एकच चेहरा camera मध्ये ठेवा."
        )

        return

    x, y, w, h = faces[0]

    face = gray[
        y:y+h,
        x:x+w
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    label, distance = (
        recognizer.predict(face)
    )

    matched_roll = (
        label_to_roll.get(label)
    )

    st.write(
        f"🔍 Face Distance: "
        f"**{distance:.2f}**"
    )

    THRESHOLD = 100

    if (
        matched_roll == roll_no
        and distance < THRESHOLD
    ):

        st.success(
            "🟢 Face Verification Successful"
        )

        success, message = mark_attendance(
            roll_no
        )

        if success:

            st.success(message)

            st.balloons()

        else:

            st.warning(message)

    else:

        st.error(
            "🔴 Face Verification Failed"
        )

        st.info(
            "💡 Good lighting वापरा आणि camera कडे सरळ बघा."
        )

    cv2.rectangle(
        frame,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Distance: {distance:.1f}",
        (x, max(y-10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    st.image(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        ),
        caption="Face Verification",
        use_container_width=True
    )


# =========================================================
# ADMIN FACE ATTENDANCE
# =========================================================

def admin_face_attendance():

    st.header(
        "📷 Admin Face Attendance"
    )

    result, error = create_recognizer()

    if error:

        st.error(error)

        return

    recognizer, label_to_roll = result

    camera = st.camera_input(
        "📷 Capture Student Face",
        key="admin_camera"
    )

    if camera is None:

        st.info(
            "Student face capture करा."
        )

        return

    image_array = np.frombuffer(
        camera.getvalue(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "❌ Image read failed."
        )

        return

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:

        st.error(
            "❌ Face detected नाही."
        )

        return

    if len(faces) > 1:

        st.warning(
            "⚠️ एका वेळी फक्त एक student."
        )

        return

    x, y, w, h = faces[0]

    face = gray[
        y:y+h,
        x:x+w
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    label, distance = (
        recognizer.predict(face)
    )

    roll_no = label_to_roll.get(
        label
    )

    st.write(
        f"🔍 Face Distance: "
        f"**{distance:.2f}**"
    )

    THRESHOLD = 100

    if (
        roll_no is not None
        and distance < THRESHOLD
    ):

        st.success(
            f"🟢 Face Verified | Roll No: {roll_no}"
        )

        success, message = mark_attendance(
            roll_no
        )

        if success:

            st.success(message)

            st.balloons()

        else:

            st.warning(message)

        text = (
            f"{roll_no} | "
            f"{distance:.1f}"
        )

    else:

        st.error(
            "🔴 Unknown Face"
        )

        text = (
            f"Unknown | "
            f"{distance:.1f}"
        )

    cv2.rectangle(
        frame,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        text,
        (x, max(y-10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    st.image(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        ),
        caption="AI Face Verification",
        use_container_width=True
    )


# =========================================================
# STUDENT REGISTRATION
# =========================================================

def student_registration():

    st.header(
        "👨‍🎓 Student Registration"
    )

    with st.form(
        "student_registration_form"
    ):

        c1, c2 = st.columns(2)

        with c1:

            name = st.text_input(
                "👤 Student Name *"
            )

            roll_no = st.text_input(
                "🔢 Roll Number *"
            )

            prn = st.text_input(
                "🆔 PRN"
            )

            email = st.text_input(
                "📧 Student Email *"
            )

        with c2:

            phone = st.text_input(
                "📱 Phone"
            )

            department = st.selectbox(
                "🏫 Department",
                [
                    "AI & Data Science",
                    "Computer Engineering",
                    "Information Technology",
                    "ENTC",
                    "Mechanical",
                    "Civil"
                ]
            )

            year = st.selectbox(
                "🎓 Year",
                [
                    "1st Year",
                    "2nd Year",
                    "3rd Year",
                    "4th Year"
                ]
            )

            photo = st.file_uploader(
                "📸 Face Photo *",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

        submit = st.form_submit_button(
            "➕ Register Student",
            use_container_width=True
        )

    if not submit:

        return

    if not name.strip():

        st.warning(
            "⚠️ Name required."
        )

        return

    if not roll_no.strip():

        st.warning(
            "⚠️ Roll Number required."
        )

        return

    if not email.strip():

        st.warning(
            "⚠️ Email required."
        )

        return

    if photo is None:

        st.warning(
            "⚠️ Face photo upload करा."
        )

        return

    roll_no = roll_no.strip()

    email = email.strip().lower()

    # Duplicate roll
    try:

        existing = (
            supabase
            .table("students")
            .select("id")
            .eq(
                "roll_no",
                roll_no
            )
            .execute()
        )

        if existing.data:

            st.error(
                "❌ Roll Number already registered."
            )

            return

    except Exception as e:

        st.error(str(e))

        return

    image_array = np.frombuffer(
        photo.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "❌ Image read failed."
        )

        return

    face, box = detect_single_face(
        image
    )

    # =====================================================
    # IMPORTANT FIX
    # =====================================================

    if is_multiple_face(face):

        st.error(
            "❌ Photo मध्ये फक्त एकच face असावा."
        )

        return

    if face is None:

        st.error(
            "❌ Face सापडला नाही."
        )

        return

    # =====================================================

    x, y, w, h = box

    face_color = image[
        y:y+h,
        x:x+w
    ]

    face_color = cv2.resize(
        face_color,
        (200, 200)
    )

    # Keep a local cache for development; cloud recognition reads from Supabase Storage.
    face_path = KNOWN_FACES_DIR / f"{roll_no}.jpg"
    if not cv2.imwrite(str(face_path), image):
        st.error("❌ Face image save failed.")
        return

    data = {

        "name":
            name.strip(),

        "roll_no":
            roll_no,

        "prn":
            prn.strip(),

        "email":
            email,

        "phone":
            phone.strip(),

        "department":
            department,

        "year":
            year
    }

    try:
        response = (
            supabase
            .table("students")
            .insert(data)
            .execute()
        )

        if not response.data:
            if face_path.exists():
                face_path.unlink()
            st.error("❌ Registration failed: database returned no student record.")
            return

        student_record = response.data[0]
        student_id = student_record.get("id")
        if student_id is None:
            st.error(
                "Student was added, but Supabase did not return an id. "
                "Ensure students.id is a generated primary key."
            )
            return

        uploaded, storage_path, upload_error = upload_face_image(
            student_id=student_id,
            image=image,
            roll_no=roll_no,
        )

        if not uploaded:
            try:
                supabase.table("students").delete().eq("id", student_id).execute()
            except Exception:
                pass
            if face_path.exists():
                face_path.unlink()
            st.error("❌ Face image upload to Supabase Storage failed.")
            st.caption(upload_error or "Unknown Storage error")
            st.info(
                "Check that the 'student-faces' bucket exists and that your Supabase "
                "key/policies allow Storage uploads."
            )
            return

        update_result = (
            supabase
            .table("students")
            .update({"face_image_paths": [storage_path]})
            .eq("id", student_id)
            .execute()
        )

        if not update_result.data:
            st.warning(
                "Student and photo uploaded, but face_image_paths could not be saved. "
                "Check the students.face_image_paths JSONB column."
            )
            st.caption(f"Uploaded Storage path: {storage_path}")
        else:
            st.success(
                f"✅ {name} registered successfully. Face photo saved to Supabase Storage."
            )

        st.image(
            cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB),
            caption="Registered Face",
            width=250,
        )

    except Exception as e:
        if face_path.exists():
            face_path.unlink()
        st.error("❌ Database registration / face upload failed.")
        st.caption(str(e))


# =========================================================
# STUDENT LIST
# =========================================================

def student_list():

    st.header(
        "📋 Student Management"
    )

    students = get_students()

    if students.empty:

        st.info(
            "No students registered."
        )

        return

    search = st.text_input(
        "🔎 Search Student",
        key="student_search"
    )

    if search:

        search = search.lower().strip()

        mask = (
            students
            .astype(str)
            .apply(
                lambda col:
                col.str.lower()
                .str.contains(
                    search,
                    na=False
                )
            )
            .any(axis=1)
        )

        students = students[mask]

    st.metric(
        "👨‍🎓 Students",
        len(students)
    )

    st.dataframe(
        students,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DAILY ATTENDANCE
# =========================================================

def attendance_list():

    st.header(
        "📊 Attendance Records"
    )

    attendance = get_attendance()

    if attendance.empty:

        st.info(
            "No attendance records."
        )

        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📋 Records",
        len(attendance)
    )

    c2.metric(
        "✅ Present",
        len(
            attendance[
                attendance["status"]
                .astype(str)
                .str.lower()
                == "present"
            ]
        )
    )

    c3.metric(
        "👨‍🎓 Students",
        attendance["roll_no"].nunique()
    )

    st.dataframe(
        attendance,
        use_container_width=True,
        hide_index=True
    )

    csv = (
        attendance
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "📥 Download Attendance CSV",
        csv,
        "attendance.csv",
        "text/csv",
        use_container_width=True
    )


# =========================================================
# MONTHLY ATTENDANCE
# =========================================================

def monthly_attendance():

    st.header(
        "📅 Advanced Monthly Attendance"
    )

    c1, c2 = st.columns(2)

    with c1:

        selected_month = st.selectbox(
            "📅 Month",
            list(range(1, 13)),
            format_func=lambda x:
                datetime(
                    2000,
                    x,
                    1
                ).strftime("%B"),
            key="monthly_month"
        )

    with c2:

        selected_year = st.number_input(
            "📆 Year",
            min_value=2020,
            max_value=2100,
            value=datetime.now().year,
            step=1,
            key="monthly_year"
        )

    month = int(selected_month)

    year = int(selected_year)

    students = get_students()

    attendance = get_attendance()

    if students.empty:

        st.warning(
            "⚠️ No students registered."
        )

        return

    if not attendance.empty:

        attendance["date"] = pd.to_datetime(
            attendance["date"],
            errors="coerce"
        )

        month_data = attendance[
            (
                attendance["date"].dt.month
                == month
            )
            &
            (
                attendance["date"].dt.year
                == year
            )
        ]

    else:

        month_data = pd.DataFrame()

    # -----------------------------------------------------
    # Working days — exclude admin holidays
    # -----------------------------------------------------

    days_in_month = calendar.monthrange(
        year,
        month
    )[1]

    today = date.today()

    if (
        year == today.year
        and month == today.month
    ):

        last_day = today.day

    elif (
        date(year, month, 1) > today
    ):

        last_day = 0

    else:

        last_day = days_in_month

    holiday_dates = set()
    holidays = get_holidays()

    if not holidays.empty and "holiday_date" in holidays.columns:
        holiday_series = pd.to_datetime(
            holidays["holiday_date"],
            errors="coerce"
        ).dropna()

        holiday_dates = {
            d.date()
            for d in holiday_series
            if d.year == year and d.month == month and d.date() <= today
        }

    total_days = max(last_day - len(holiday_dates), 0)

    report = []

    for _, student in students.iterrows():

        roll_no = str(
            student.get(
                "roll_no",
                ""
            )
        ).strip()

        name = str(
            student.get(
                "name",
                ""
            )
        )

        department = str(
            student.get(
                "department",
                ""
            )
        )

        if (
            total_days == 0
            or month_data.empty
        ):

            present = 0

        else:

            student_data = month_data[
                month_data["roll_no"]
                .astype(str)
                .str.strip()
                == roll_no
            ]

            present = len(
                student_data[
                    student_data["status"]
                    .astype(str)
                    .str.lower()
                    == "present"
                ]
            )

        absent = max(
            total_days - present,
            0
        )

        percentage = (
            present
            / total_days
            * 100
            if total_days
            else 0
        )

        if percentage >= 75:

            status = "🟢 Good"

        elif percentage >= 60:

            status = "🟡 Warning"

        else:

            status = "🔴 Low"

        report.append({

            "Roll No":
                roll_no,

            "Name":
                name,

            "Department":
                department,

            "Total Days":
                total_days,

            "Present":
                present,

            "Absent":
                absent,

            "Attendance %":
                round(
                    percentage,
                    2
                ),

            "Status":
                status
        })

    report_df = pd.DataFrame(
        report
    )

    month_name = datetime(
        year,
        month,
        1
    ).strftime("%B %Y")

    st.subheader(
        f"📊 {month_name}"
    )

    if holiday_dates:
        st.info(
            f"🏖️ Holidays excluded from working days: {len(holiday_dates)}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total_students = len(
        report_df
    )

    total_present = int(
        report_df["Present"].sum()
    )

    total_absent = int(
        report_df["Absent"].sum()
    )

    average = (
        report_df["Attendance %"].mean()
        if not report_df.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👨‍🎓 Students",
        total_students
    )

    c2.metric(
        "✅ Present",
        total_present
    )

    c3.metric(
        "❌ Absent",
        total_absent
    )

    c4.metric(
        "📊 Average",
        f"{average:.1f}%"
    )

    st.divider()

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    search = st.text_input(
        "🔎 Search Roll No / Name",
        key="monthly_search"
    )

    display_df = report_df.copy()

    if search:

        search = search.lower().strip()

        mask = (
            display_df
            .astype(str)
            .apply(
                lambda col:
                col.str.lower()
                .str.contains(
                    search,
                    na=False
                )
            )
            .any(axis=1)
        )

        display_df = display_df[mask]

    # -----------------------------------------------------
    # Table
    # -----------------------------------------------------

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Chart
    # -----------------------------------------------------

    st.subheader(
        "📈 Student-wise Attendance"
    )

    if not display_df.empty:

        chart = display_df[
            [
                "Name",
                "Attendance %"
            ]
        ].copy()

        chart = chart.set_index(
            "Name"
        )

        st.bar_chart(
            chart
        )

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv = (
        report_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "📥 Download Monthly Report",
        csv,
        f"monthly_attendance_{year}_{month:02d}.csv",
        "text/csv",
        use_container_width=True
    )


# =========================================================
# GRADE
# =========================================================

def calculate_grade(percentage):

    percentage = float(
        percentage
    )

    if percentage >= 90:
        return "A+"

    if percentage >= 80:
        return "A"

    if percentage >= 70:
        return "B+"

    if percentage >= 60:
        return "B"

    if percentage >= 50:
        return "C"

    if percentage >= 40:
        return "D"

    return "F"


def calculate_result(percentage):

    return (
        "PASS"
        if float(percentage) >= 40
        else "FAIL"
    )


# =========================================================
# ADD MARKS
# =========================================================

def add_marks():

    st.header(
        "📝 Add Student Marks"
    )

    students = get_students()

    if students.empty:

        st.warning(
            "No students registered."
        )

        return

    students = students.copy()

    students["display"] = (
        students["roll_no"]
        .astype(str)
        + " - "
        + students["name"]
        .astype(str)
    )

    selected = st.selectbox(
        "👨‍🎓 Select Student",
        students["display"].tolist(),
        key="marks_student"
    )

    student = students[
        students["display"]
        == selected
    ].iloc[0]

    st.info(
        f"👤 **{student['name']}** | "
        f"Roll No: **{student['roll_no']}**"
    )

    c1, c2 = st.columns(2)

    with c1:

        semester = st.selectbox(
            "🎓 Semester",
            [
                "Semester 1",
                "Semester 2",
                "Semester 3",
                "Semester 4",
                "Semester 5",
                "Semester 6",
                "Semester 7",
                "Semester 8"
            ],
            key="marks_semester"
        )

    with c2:

        subject = st.text_input(
            "📚 Subject",
            key="marks_subject"
        )

    c1, c2 = st.columns(2)

    with c1:

        internal = st.number_input(
            "Internal /20",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            key="internal_marks"
        )

        practical = st.number_input(
            "Practical /20",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            key="practical_marks"
        )

    with c2:

        assignment = st.number_input(
            "Assignment /10",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0,
            key="assignment_marks"
        )

        end_sem = st.number_input(
            "End Semester /50",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            key="end_sem_marks"
        )

    total = (
        internal
        + practical
        + assignment
        + end_sem
    )

    percentage = total

    grade = calculate_grade(
        percentage
    )

    result = calculate_result(
        percentage
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total",
        f"{total:.0f}/100"
    )

    c2.metric(
        "Percentage",
        f"{percentage:.1f}%"
    )

    c3.metric(
        "Grade",
        grade
    )

    c4.metric(
        "Result",
        result
    )

    if st.button(
        "💾 Save Marks",
        use_container_width=True,
        key="save_marks_button"
    ):

        if not subject.strip():

            st.warning(
                "⚠️ Subject enter करा."
            )

            return

        try:

            student_id = student.get(
                "id"
            )

            data = {

                "student_id":
                    student_id,

                "roll_no":
                    str(
                        student["roll_no"]
                    ),

                "student_name":
                    str(
                        student["name"]
                    ),

                "semester":
                    semester,

                "subject":
                    subject.strip(),

                "internal":
                    float(internal),

                "practical":
                    float(practical),

                "assignment":
                    float(assignment),

                "end_sem":
                    float(end_sem),

                "total":
                    float(total),

                "percentage":
                    float(percentage),

                "grade":
                    grade,

                "result":
                    result
            }

            response = (
                supabase
                .table("marks")
                .insert(data)
                .execute()
            )

            if response.data:

                st.success(
                    "✅ Marks saved successfully!"
                )

                st.balloons()

        except Exception as e:

            st.error(
                "❌ Marks save failed."
            )

            st.caption(
                str(e)
            )


# =========================================================
# MARKS RECORDS
# =========================================================

def marks_records():

    st.header(
        "📚 Marks Management"
    )

    marks = get_marks()

    if marks.empty:

        st.info(
            "No marks records."
        )

        return

    percentage = pd.to_numeric(
        marks["percentage"],
        errors="coerce"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📚 Records",
        len(marks)
    )

    c2.metric(
        "👨‍🎓 Students",
        marks["roll_no"].nunique()
    )

    c3.metric(
        "📊 Average",
        f"{percentage.mean():.1f}%"
    )

    c4.metric(
        "❌ Failed",
        len(
            marks[
                marks["result"]
                .astype(str)
                .str.upper()
                == "FAIL"
            ]
        )
    )

    search = st.text_input(
        "🔎 Search",
        key="marks_search"
    )

    if search:

        search = search.lower().strip()

        mask = (
            marks
            .astype(str)
            .apply(
                lambda col:
                col.str.lower()
                .str.contains(
                    search,
                    na=False
                )
            )
            .any(axis=1)
        )

        marks = marks[mask]

    columns = [

        "roll_no",

        "student_name",

        "semester",

        "subject",

        "internal",

        "practical",

        "assignment",

        "end_sem",

        "total",

        "percentage",

        "grade",

        "result"
    ]

    available = [
        c
        for c in columns
        if c in marks.columns
    ]

    st.dataframe(
        marks[available],
        use_container_width=True,
        hide_index=True
    )

    csv = (
        marks[available]
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "📥 Download Marks CSV",
        csv,
        "marks.csv",
        "text/csv",
        use_container_width=True
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

def admin_dashboard():

    st.header(
        "👨‍💼 Admin Dashboard"
    )

    students = get_students()

    attendance = get_attendance()

    total_students = len(
        students
    )

    today = date.today().isoformat()

    if not attendance.empty:

        today_data = attendance[
            attendance["date"]
            .astype(str)
            .str[:10]
            == today
        ]

    else:

        today_data = pd.DataFrame()

    today_holiday, holiday_name = is_holiday(date.today())

    present = len(
        today_data
    )

    absent = (
        0
        if today_holiday
        else max(total_students - present, 0)
    )

    percentage = (
        present
        / total_students
        * 100
        if total_students
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    c2.metric(
        "✅ Present",
        present
    )

    c3.metric(
        "❌ Absent",
        absent
    )

    c4.metric(
        "📊 Today",
        f"{percentage:.1f}%"
    )

    st.divider()

    if today_holiday:
        st.success(
            f"🏖️ Today is a holiday — {holiday_name}. Attendance is not required."
        )

    st.subheader(
        "📅 Today's Attendance"
    )

    if today_data.empty:

        st.info(
            "आजची attendance नाही."
        )

    else:

        st.dataframe(
            today_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# STUDENT HOLIDAYS
# =========================================================

def student_holidays():

    st.header("🏖️ Holidays")
    st.caption("College holidays announced by Admin")

    holidays = get_holidays()

    if holidays.empty:
        st.info("📅 No holidays have been announced yet.")
        return

    if "holiday_date" not in holidays.columns:
        st.error("❌ Holiday date column is missing.")
        return

    holidays["holiday_date"] = pd.to_datetime(
        holidays["holiday_date"], errors="coerce"
    )
    holidays = holidays.dropna(subset=["holiday_date"]).sort_values("holiday_date")

    today = pd.Timestamp(date.today())
    upcoming = holidays[holidays["holiday_date"] >= today].copy()

    if upcoming.empty:
        st.success("📅 No upcoming holidays currently announced.")
    else:
        st.subheader("📅 Upcoming Holidays")
        for _, row in upcoming.iterrows():
            hdate = row["holiday_date"].strftime("%d %B %Y")
            hname = str(row.get("holiday_name", "Holiday"))
            st.success(f"🏖️ **{hname}**  —  {hdate}")

    st.divider()
    st.subheader("📋 All Announced Holidays")

    display = holidays.copy()
    display["holiday_date"] = display["holiday_date"].dt.strftime("%d-%m-%Y")

    cols = [c for c in ["holiday_date", "holiday_name"] if c in display.columns]
    display = display[cols].rename(columns={
        "holiday_date": "Date",
        "holiday_name": "Holiday"
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# STUDENT DASHBOARD
# =========================================================

def student_dashboard():

    student = get_logged_student()

    if student is None:

        st.warning(
            "❌ Student registration record सापडला नाही."
        )

        return

    roll_no = str(
        student["roll_no"]
    ).strip()

    st.header(
        "👨‍🎓 Student Dashboard"
    )

    st.subheader(
        f"👋 Welcome, {student['name']}"
    )

    c1, c2, c3 = st.columns(3)

    c1.write(
        f"**Roll No:** {roll_no}"
    )

    c2.write(
        f"**Email:** {student.get('email', '')}"
    )

    c3.write(
        f"**Department:** "
        f"{student.get('department', '')}"
    )

    st.divider()

    # =====================================================
    # ATTENDANCE
    # =====================================================

    st.subheader(
        "📊 My Attendance"
    )

    attendance = get_attendance()

    if not attendance.empty:

        my_attendance = attendance[
            attendance["roll_no"]
            .astype(str)
            .str.strip()
            == roll_no
        ].copy()

    else:

        my_attendance = pd.DataFrame()

    total_days = len(
        my_attendance
    )

    if not my_attendance.empty:

        present_days = len(
            my_attendance[
                my_attendance["status"]
                .astype(str)
                .str.lower()
                == "present"
            ]
        )

    else:

        present_days = 0

    absent_days = max(
        total_days - present_days,
        0
    )

    attendance_percentage = (
        present_days
        / total_days
        * 100
        if total_days
        else 0
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "📅 Total",
        total_days
    )

    a2.metric(
        "✅ Present",
        present_days
    )

    a3.metric(
        "❌ Absent",
        absent_days
    )

    a4.metric(
        "📊 Percentage",
        f"{attendance_percentage:.1f}%"
    )

    if attendance_percentage < 75:

        st.warning(
            "⚠️ Attendance below 75%."
        )

    else:

        st.success(
            "✅ Attendance above 75%."
        )

    if not my_attendance.empty:

        st.dataframe(
            my_attendance,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # MARKS
    # =====================================================

    st.divider()

    st.subheader(
        "📚 My Marks"
    )

    marks = get_marks()

    if not marks.empty:

        my_marks = marks[
            marks["roll_no"]
            .astype(str)
            .str.strip()
            == roll_no
        ].copy()

    else:

        my_marks = pd.DataFrame()

    if my_marks.empty:

        st.info(
            "📚 Marks available नाहीत."
        )

        return

    percentage = pd.to_numeric(
        my_marks["percentage"],
        errors="coerce"
    )

    avg = percentage.mean()

    highest = percentage.max()

    passed = len(
        my_marks[
            my_marks["result"]
            .astype(str)
            .str.upper()
            == "PASS"
        ]
    )

    failed = len(
        my_marks[
            my_marks["result"]
            .astype(str)
            .str.upper()
            == "FAIL"
        ]
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "📊 Average",
        f"{avg:.1f}%"
    )

    m2.metric(
        "🏆 Highest",
        f"{highest:.1f}%"
    )

    m3.metric(
        "✅ Passed",
        passed
    )

    m4.metric(
        "❌ Failed",
        failed
    )

    columns = [

        "semester",

        "subject",

        "internal",

        "practical",

        "assignment",

        "end_sem",

        "total",

        "percentage",

        "grade",

        "result"
    ]

    available = [
        c
        for c in columns
        if c in my_marks.columns
    ]

    st.dataframe(
        my_marks[available],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# LOGOUT
# =========================================================

def logout():

    try:

        supabase.auth.sign_out()

    except Exception:

        pass

    st.session_state.logged_in = False

    st.session_state.user = None

    st.session_state.role = None

    st.session_state.page = None

    st.rerun()


# =========================================================
# LOGIN / SIGNUP
# =========================================================

if not st.session_state.logged_in:

    login_tab, signup_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Sign Up"
        ]
    )

    with login_tab:

        login_page()

    with signup_tab:

        signup_page()


# =========================================================
# LOGGED-IN APPLICATION
# =========================================================

else:

    user = st.session_state.user

    role = st.session_state.role

    # =====================================================
    # ADMIN
    # =====================================================

    if role == "admin":

        st.sidebar.title(
            "👨‍💼 Admin Panel"
        )

        st.sidebar.success(
            f"👤 {user.email}"
        )

        page = st.sidebar.radio(
            "Navigation",
            [

                "📊 Admin Dashboard",

                "👨‍🎓 Student Registration",

                "📋 Student List",

                "📷 Face Attendance",

                "📊 Attendance Records",

                "📅 Monthly Attendance",

                "📝 Add Marks",

                "📚 Marks Records",

                "🏖️ Holiday Management"
            ],
            key="admin_navigation"
        )

        st.sidebar.divider()

        if st.sidebar.button(
            "🚪 Logout",
            use_container_width=True,
            key="admin_logout"
        ):

            logout()

        if page == "📊 Admin Dashboard":

            admin_dashboard()

        elif page == "👨‍🎓 Student Registration":

            student_registration()

        elif page == "📋 Student List":

            student_list()

        elif page == "📷 Face Attendance":

            admin_face_attendance()

        elif page == "📊 Attendance Records":

            attendance_list()

        elif page == "📅 Monthly Attendance":

            monthly_attendance()

        elif page == "📝 Add Marks":

            add_marks()

        elif page == "📚 Marks Records":

            marks_records()

        elif page == "🏖️ Holiday Management":

            holiday_management()

    # =====================================================
    # STUDENT
    # =====================================================

    elif role == "student":

        st.sidebar.title(
            "👨‍🎓 Student Panel"
        )

        st.sidebar.success(
            f"👤 {user.email}"
        )

        page = st.sidebar.radio(
            "Navigation",
            [

                "📊 Student Dashboard",

                "🏖️ Holidays",

                "📷 Mark My Attendance"
            ],
            key="student_navigation"
        )

        st.sidebar.divider()

        if st.sidebar.button(
            "🚪 Logout",
            use_container_width=True,
            key="student_logout"
        ):

            logout()

        if page == "📊 Student Dashboard":

            student_dashboard()

        elif page == "🏖️ Holidays":

            student_holidays()

        elif page == "📷 Mark My Attendance":

            student_face_attendance()

    else:

        st.error(
            "❌ Unauthorized role."
        )

        logout()
