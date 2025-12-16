"""
End-to-end tests for essay HTML pages using Playwright.

These tests verify that:
1. The essay HTML renders correctly in a browser
2. All D3.js charts render without JavaScript errors
3. Scrollytelling interactions work properly
4. Screenshots are captured for visual regression testing
"""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

# Test configuration
PROJECT_ROOT = Path(__file__).parent.parent
ESSAY_OUTPUT_DIR = PROJECT_ROOT / "output" / "essays"
SCREENSHOT_DIR = PROJECT_ROOT / "output" / "screenshots"
VIDEO_DIR = PROJECT_ROOT / "output" / "videos"


@pytest.fixture(scope="session", autouse=True)
def setup_directories():
    """Create output directories for screenshots and videos."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def essay_path() -> Path:
    """Get the path to the generated essay HTML file."""
    essay_file = ESSAY_OUTPUT_DIR / "index.html"
    if not essay_file.exists():
        pytest.skip(f"Essay file not found: {essay_file}. Run generate_essay.py first.")
    return essay_file


@pytest.fixture
def console_errors(page: Page) -> list[str]:
    """Collect JavaScript console errors during page interaction."""
    errors: list[str] = []

    def handle_console(msg):
        if msg.type == "error":
            errors.append(f"[{msg.type}] {msg.text}")

    page.on("console", handle_console)
    return errors


@pytest.fixture
def page_errors(page: Page) -> list[str]:
    """Collect page-level errors (uncaught exceptions)."""
    errors: list[str] = []

    def handle_error(error):
        errors.append(f"Page error: {error}")

    page.on("pageerror", handle_error)
    return errors


class TestEssayPageLoad:
    """Tests for basic page loading and structure."""

    def test_page_loads_successfully(self, page: Page, essay_path: Path):
        """Essay page should load without network errors."""
        response = page.goto(f"file://{essay_path}")
        assert response is not None
        assert response.ok, f"Page failed to load: {response.status}"

    def test_page_title_present(self, page: Page, essay_path: Path):
        """Page should have a title."""
        page.goto(f"file://{essay_path}")
        title = page.title()
        assert title, "Page should have a title"
        assert "Customer" in title or "Segmentation" in title or "Story" in title

    def test_no_javascript_errors_on_load(
        self, page: Page, essay_path: Path, console_errors: list, page_errors: list
    ):
        """Page should load without JavaScript errors."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        # Wait a bit for any async JS to execute
        page.wait_for_timeout(2000)

        # Filter out non-critical errors (e.g., favicon 404)
        critical_errors = [
            e for e in console_errors + page_errors
            if "favicon" not in e.lower() and "404" not in e
        ]

        assert not critical_errors, f"JavaScript errors found: {critical_errors}"

    def test_d3_library_loaded(self, page: Page, essay_path: Path):
        """D3.js library should be loaded and available."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        d3_loaded = page.evaluate("typeof d3 !== 'undefined'")
        assert d3_loaded, "D3.js library should be loaded"

    def test_scrollama_library_loaded(self, page: Page, essay_path: Path):
        """Scrollama library should be loaded and available."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        scrollama_loaded = page.evaluate("typeof scrollama !== 'undefined'")
        assert scrollama_loaded, "Scrollama library should be loaded"


class TestEssaySections:
    """Tests for individual essay sections."""

    def test_hero_section_visible(self, page: Page, essay_path: Path):
        """Hero section should be visible on page load."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        hero = page.locator(".hero, .intro, header, [class*='hero']").first
        expect(hero).to_be_visible(timeout=5000)

    def test_essay_sections_exist(self, page: Page, essay_path: Path):
        """Multiple essay sections should exist."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        sections = page.locator("section, .essay-section, [class*='section']")
        count = sections.count()
        assert count >= 4, f"Expected at least 4 sections, found {count}"

    def test_charts_render(self, page: Page, essay_path: Path):
        """Charts should render as SVG elements."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)  # Wait for chart animations

        # Check for SVG elements (D3 charts)
        svgs = page.locator("svg")
        svg_count = svgs.count()
        assert svg_count >= 1, f"Expected at least 1 SVG chart, found {svg_count}"

    def test_chart_containers_have_content(self, page: Page, essay_path: Path):
        """Chart containers should have rendered content."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Check for chart containers with actual content
        chart_containers = page.locator("[id^='chart-']")

        for i in range(min(chart_containers.count(), 5)):
            container = chart_containers.nth(i)
            # Container should have some content (SVG, canvas, or child elements)
            inner_html = container.inner_html()
            assert len(inner_html) > 10, f"Chart container {i} appears empty"


class TestScrollytelling:
    """Tests for scrollytelling functionality."""

    def test_scroll_triggers_exist(self, page: Page, essay_path: Path):
        """Scroll trigger elements should exist."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        # Look for scroll steps or triggers
        triggers = page.locator(".step, .scroll-step, [class*='step'], [class*='trigger']")
        count = triggers.count()
        assert count >= 1, f"Expected scroll triggers, found {count}"

    def test_scrolling_changes_content(self, page: Page, essay_path: Path):
        """Scrolling should trigger visual changes."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # Take screenshot before scrolling
        initial_screenshot = page.screenshot()

        # Scroll down significantly
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        page.wait_for_timeout(1000)

        # Take screenshot after scrolling
        scrolled_screenshot = page.screenshot()

        # Screenshots should be different (page content changed)
        assert initial_screenshot != scrolled_screenshot, "Page content should change on scroll"


class TestScreenshots:
    """Tests that capture screenshots for visual review."""

    def test_capture_full_page_screenshot(self, page: Page, essay_path: Path):
        """Capture a full-page screenshot."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)  # Wait for animations

        screenshot_path = SCREENSHOT_DIR / "full_page.png"
        page.screenshot(path=str(screenshot_path), full_page=True)

        assert screenshot_path.exists(), "Full page screenshot should be saved"
        assert screenshot_path.stat().st_size > 10000, "Screenshot should have content"

    def test_capture_hero_section(self, page: Page, essay_path: Path):
        """Capture screenshot of hero/intro section."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        screenshot_path = SCREENSHOT_DIR / "hero_section.png"
        page.screenshot(path=str(screenshot_path))

        assert screenshot_path.exists()

    def test_capture_each_section_screenshot(self, page: Page, essay_path: Path):
        """Capture screenshots of each major section."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        sections = page.locator("section")
        section_count = sections.count()

        captured = 0
        for i in range(min(section_count, 10)):  # Cap at 10 sections
            section = sections.nth(i)
            try:
                section.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)  # Wait for scroll animation

                screenshot_path = SCREENSHOT_DIR / f"section_{i:02d}.png"
                section.screenshot(path=str(screenshot_path))
                captured += 1
            except Exception:
                # Section might not be visible or have zero size
                continue

        assert captured >= 1, "Should capture at least one section screenshot"

    def test_capture_chart_screenshots(self, page: Page, essay_path: Path):
        """Capture screenshots of each chart."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        chart_containers = page.locator("[id^='chart-']")
        chart_count = chart_containers.count()

        captured = 0
        for i in range(min(chart_count, 10)):
            container = chart_containers.nth(i)
            try:
                container.scroll_into_view_if_needed()
                page.wait_for_timeout(500)

                chart_id = container.get_attribute("id") or f"chart_{i}"
                screenshot_path = SCREENSHOT_DIR / f"{chart_id}.png"
                container.screenshot(path=str(screenshot_path))
                captured += 1
            except Exception:
                continue

        assert captured >= 1, "Should capture at least one chart screenshot"


class TestVideoRecording:
    """Tests with video recording enabled."""

    @pytest.fixture
    def browser_context_args(self, browser_context_args):
        """Configure browser context to record video."""
        return {
            **browser_context_args,
            "record_video_dir": str(VIDEO_DIR),
            "record_video_size": {"width": 1280, "height": 720},
        }

    def test_record_scroll_through_essay(self, page: Page, essay_path: Path):
        """Record video of scrolling through the entire essay."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # Get total page height
        total_height = page.evaluate("document.body.scrollHeight")
        viewport_height = page.evaluate("window.innerHeight")

        # Scroll through the page in steps
        scroll_step = viewport_height * 0.8
        current_position = 0

        while current_position < total_height:
            page.evaluate(f"window.scrollTo(0, {current_position})")
            page.wait_for_timeout(800)  # Pause for video
            current_position += scroll_step

        # Scroll to bottom
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)

        # Scroll back to top
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1000)


class TestDataIntegrity:
    """Tests that verify data is correctly displayed."""

    def test_customer_count_displayed(self, page: Page, essay_path: Path):
        """Customer count should be displayed somewhere on the page."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        page_text = page.inner_text("body")

        # Should mention customer count (1,000 or 1000)
        has_customer_count = bool(
            re.search(r"1[,.]?000\s*customers?", page_text, re.IGNORECASE) or
            re.search(r"customers?[:\s]+1[,.]?000", page_text, re.IGNORECASE)
        )

        assert has_customer_count, "Page should display customer count"

    def test_percentages_displayed(self, page: Page, essay_path: Path):
        """Percentage values should be displayed."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        page_text = page.inner_text("body")

        # Should contain percentage values
        percentages = re.findall(r"\d+\.?\d*%", page_text)
        assert len(percentages) >= 3, f"Expected multiple percentages, found {len(percentages)}"

    def test_currency_values_displayed(self, page: Page, essay_path: Path):
        """Currency/revenue values should be displayed."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        page_text = page.inner_text("body")

        # Should contain currency values
        has_currency = bool(
            re.search(r"\$[\d,]+", page_text) or
            re.search(r"revenue|clv|ltv|value", page_text, re.IGNORECASE)
        )

        assert has_currency, "Page should display currency/value information"


class TestAccessibility:
    """Basic accessibility tests."""

    def test_page_has_headings(self, page: Page, essay_path: Path):
        """Page should have proper heading structure."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        h1_count = page.locator("h1").count()
        h2_count = page.locator("h2").count()

        assert h1_count >= 1, "Page should have at least one H1"
        assert h2_count >= 1, "Page should have at least one H2"

    def test_images_have_alt_text(self, page: Page, essay_path: Path):
        """Images should have alt text (if any exist)."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        images = page.locator("img")
        img_count = images.count()

        if img_count > 0:
            for i in range(img_count):
                img = images.nth(i)
                alt = img.get_attribute("alt")
                # Alt can be empty string for decorative images, but should exist
                assert alt is not None, f"Image {i} should have alt attribute"

    def test_links_are_accessible(self, page: Page, essay_path: Path):
        """Links should have accessible text."""
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")

        links = page.locator("a")
        link_count = links.count()

        for i in range(min(link_count, 10)):
            link = links.nth(i)
            text = link.inner_text().strip()
            aria_label = link.get_attribute("aria-label")

            # Link should have either text content or aria-label
            assert text or aria_label, f"Link {i} should have accessible text"


class TestResponsiveness:
    """Tests for responsive design."""

    @pytest.mark.parametrize("viewport", [
        {"width": 1920, "height": 1080, "name": "desktop"},
        {"width": 1024, "height": 768, "name": "tablet"},
        {"width": 375, "height": 667, "name": "mobile"},
    ])
    def test_renders_at_viewport_size(self, page: Page, essay_path: Path, viewport: dict):
        """Essay should render at different viewport sizes."""
        page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
        page.goto(f"file://{essay_path}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        screenshot_path = SCREENSHOT_DIR / f"viewport_{viewport['name']}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)

        # Page should have content
        body_height = page.evaluate("document.body.scrollHeight")
        assert body_height > viewport["height"], "Page should have scrollable content"


# Pytest configuration for video recording
def pytest_configure(config):
    """Configure pytest for Playwright."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
