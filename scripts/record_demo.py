"""
Scripted Playwright walkthrough of the hypertension presentation app,
recorded to a vertical (4:5, mobile-friendly) video for a LinkedIn demo.
Opens on the discussed HIGH-risk patient (preloaded defaults), then flips
to the discussed LOW-risk patient. Outputs .webm under /tmp; a separate
ffmpeg step trims + converts to MP4.

Run: .venv/bin/python scripts/record_demo.py
"""
import asyncio
import os
from playwright.async_api import async_playwright

URL = "http://localhost:8501/?embed=true"   # embed=true strips Streamlit chrome
OUT_DIR = "/tmp/htn_demo_video"
VW, VH = 1080, 1350                          # 4:5 portrait for LinkedIn mobile feed

# discussed LOW-risk patient (the swing target)
LOW = {
    "Age (years)": 28, "Height (cm)": 168, "Weight (kg)": 60,
    "Serum Creatinine (mg/dL)": 0.7, "Uric Acid (mg/dL)": 4.0,
    "Serum Albumin (g/dL)": 4.5, "Blood Urea Nitrogen (mg/dL)": 10,
    "Urine Albumin (mg/L)": 5, "Urine Creatinine (mg/dL)": 120,
}


# Streamlit scrolls an inner container, NOT the window. In embed mode the
# real scroller is section[data-testid="stMain"]; window.scrollTo is a no-op.
SCROLLER = 'section[data-testid="stMain"]'


async def smooth_scroll_to(page, y, duration=2000, fps=50):
    """Python-paced incremental scroll of the Streamlit container. Driving it
    step-by-step (rather than requestAnimationFrame inside the page, which gets
    throttled in headless recording) guarantees the recorder captures every
    intermediate frame, so the motion is genuinely smooth and continuous."""
    if y is None:
        return
    start = await page.evaluate(f"document.querySelector('{SCROLLER}').scrollTop")
    steps = max(int(duration / 1000 * fps), 1)
    frame_ms = int(1000 / fps)
    for i in range(1, steps + 1):
        p = i / steps
        e = 4 * p * p * p if p < 0.5 else 1 - pow(-2 * p + 2, 3) / 2  # easeInOutCubic
        await page.evaluate(
            f"(v) => {{ document.querySelector('{SCROLLER}').scrollTop = v; }}",
            start + (y - start) * e,
        )
        await page.wait_for_timeout(frame_ms)


async def y_of(page, text, offset=70):
    return await page.evaluate(
        """([t, off, sel]) => {
            const sc = document.querySelector(sel);
            const els = [...sc.querySelectorAll('h1,h2,h3,h4,p,span,div')];
            const el = els.find(e => e.textContent.trim().startsWith(t));
            if (!el) return null;
            return el.getBoundingClientRect().top - sc.getBoundingClientRect().top
                   + sc.scrollTop - off;
        }""",
        [text, offset, SCROLLER],
    )


async def scroll_to_text(page, text, offset=70, duration=2000):
    y = await y_of(page, text, offset)
    if y is not None:
        await smooth_scroll_to(page, max(y, 0), duration)
    return y


async def set_num(page, label, value, settle=650):
    container = page.locator("div[data-testid='stNumberInput']").filter(has_text=label).first
    inp = container.locator("input")
    await inp.scroll_into_view_if_needed()
    await page.wait_for_timeout(120)
    await inp.click()
    await inp.fill(str(value))
    await inp.press("Enter")
    await page.wait_for_timeout(settle)


async def set_select(page, label, option, settle=650):
    container = page.locator("div[data-testid='stSelectbox']").filter(has_text=label).first
    await container.scroll_into_view_if_needed()
    await page.wait_for_timeout(120)
    await container.locator("div[data-baseweb='select']").click()
    await page.wait_for_timeout(250)
    try:
        await page.get_by_role("option", name=option, exact=True).first.click(timeout=2500)
    except Exception:
        await page.locator("li[role='option']", has_text=option).first.click()
    await page.wait_for_timeout(settle)


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": VW, "height": VH},
            record_video_dir=OUT_DIR,
            record_video_size={"width": VW, "height": VH},
            device_scale_factor=2,
        )
        page = await context.new_page()
        await page.goto(URL, wait_until="domcontentloaded")
        await page.wait_for_selector(".hero h1", timeout=30000)
        await page.wait_for_timeout(1500)  # brief settle (trimmed later)

        # ── Hero glance, then bring the pipeline into view smoothly ──
        await page.wait_for_timeout(800)
        await scroll_to_text(page, "Methodology", offset=40, duration=3000)
        try:                                                   # play the pipeline animation (the part you like)
            await page.get_by_text("Play", exact=True).first.click(timeout=2500)
        except Exception:
            pass
        await page.wait_for_timeout(2200)                      # watch it animate — then keep moving, no dwell

        # ── ONE continuous, slow, eased scroll straight to the calculator ──
        #    (glides past the results/timeline without ever stopping)
        calc_y = await y_of(page, "Prototype Risk Calculator", offset=30)
        await smooth_scroll_to(page, calc_y, duration=6000)
        await page.wait_for_timeout(3500)                      # hold on HIGH (~96%)

        # ── Enter the discussed LOW-risk patient ──
        await set_select(page, "Sex", "Female")
        await set_select(page, "Race / Ethnicity", "Non-Hispanic White")
        for label, value in LOW.items():
            await set_num(page, label, value)

        # scroll back up so the probability card + contributions are framed
        await scroll_to_text(page, "Prototype Risk Calculator", offset=30, duration=1800)
        await page.wait_for_timeout(4500)                      # hold on LOW (~6%)

        await context.close()
        await browser.close()

        vids = [f for f in os.listdir(OUT_DIR) if f.endswith(".webm")]
        vids.sort(key=lambda f: os.path.getmtime(os.path.join(OUT_DIR, f)))
        print(os.path.join(OUT_DIR, vids[-1]) if vids else "NO_VIDEO")


if __name__ == "__main__":
    asyncio.run(main())
