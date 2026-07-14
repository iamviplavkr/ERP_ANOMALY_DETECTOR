"""
tests/frontend/test_dashboard_auth.py
─────────────────────────────────────────────────────────────────
Unit and integration tests for Streamlit frontend login logic.
Mocks Streamlit APIs to test the `do_login` function behavior
under connection errors, 401 Unauthorized, 403 Forbidden, and 200 Success.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
import requests

# 1. Setup a highly robust mock streamlit module to prevent execution errors when importing dashboard.py
mock_st = MagicMock()
mock_st.session_state = {"authenticated": False}

# Helper to mock columns to return correct number of mocks so unpacking does not fail
def mock_columns(spec):
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    elif isinstance(spec, list):
        return [MagicMock() for _ in range(len(spec))]
    return [MagicMock(), MagicMock(), MagicMock()]

mock_st.columns = mock_columns

# Mock cache decorator to act as a pass-through identity decorator
def identity_decorator(func):
    return func

mock_st.cache_resource = identity_decorator

# Make radio return a specific string to avoid matching other execution blocks
mock_st.sidebar = MagicMock()
mock_st.sidebar.radio.return_value = "📊 Dashboard"

# file_uploader returns None so the file upload execution block is skipped
mock_st.file_uploader.return_value = None

# Register our mock streamlit in sys.modules
sys.modules['streamlit'] = mock_st

# 2. Import the frontend functions after mocking streamlit is fully registered
from frontend.dashboard import do_login, API_BASE_URL


class TestDashboardAuth:
    """Test suite verifying do_login response mapping states."""

    @patch("requests.post")
    def test_do_login_success(self, mock_post):
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "mock_access",
            "refresh_token": "mock_refresh",
            "username": "admin",
            "role": "Admin"
        }
        mock_post.return_value = mock_response

        # Clear state
        mock_st.session_state.clear()
        mock_st.session_state["authenticated"] = False

        # Run login
        success, error_msg = do_login("admin", "password123")

        assert success is True
        assert error_msg is None
        assert mock_st.session_state["authenticated"] is True
        assert mock_st.session_state["access_token"] == "mock_access"
        assert mock_st.session_state["username"] == "admin"
        assert mock_st.session_state["role"] == "Admin"

    @patch("requests.post")
    def test_do_login_invalid_credentials(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        success, error_msg = do_login("admin", "wrong_password")

        assert success is False
        assert error_msg == "Invalid username or password."

    @patch("requests.post")
    def test_do_login_forbidden(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_post.return_value = mock_response

        success, error_msg = do_login("deactivated", "password123")

        assert success is False
        assert error_msg == "Account is deactivated or forbidden."

    @patch("requests.post")
    def test_do_login_connection_error(self, mock_post):
        # Simulate connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        success, error_msg = do_login("admin", "password123")

        assert success is False
        assert "Cannot connect to backend API" in error_msg

    @patch("requests.post")
    def test_do_login_unexpected_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        success, error_msg = do_login("admin", "password123")

        assert success is False
        assert "Unexpected response from backend API" in error_msg


class TestDashboardAuthIntegration:
    """Integration test verifying login works E2E against a real running FastAPI instance."""

    def test_do_login_e2e_against_testclient(self):
        # We test do_login against our actual backend using a patch to requests.post
        # that redirects requests directly to the FastAPI TestClient
        from starlette.testclient import TestClient
        from backend.main import app

        client = TestClient(app)

        def mock_post_to_testclient(url, json, timeout=None):
            # Route requests to FastAPI TestClient
            relative_path = url.replace(API_BASE_URL, "")
            return client.post(relative_path, json=json)

        # Clear state
        mock_st.session_state.clear()
        mock_st.session_state["authenticated"] = False

        with patch("requests.post", side_effect=mock_post_to_testclient):
            # Test success case
            success, error_msg = do_login("admin", "password123")
            assert success is True
            assert error_msg is None
            assert mock_st.session_state["role"] == "Admin"

            # Test failure case
            mock_st.session_state.clear()
            mock_st.session_state["authenticated"] = False
            success, error_msg = do_login("admin", "wrong")
            assert success is False
            assert error_msg == "Invalid username or password."
