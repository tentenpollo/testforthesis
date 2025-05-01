import streamlit as st
from user_management import authenticate_user, register_user, user_exists

def show_login_page():
    """Display the login page and handle authentication"""
    st.title("🍎 Fruit Ripeness Detection System")
    st.subheader("Login")
    
    # Check if the user is already logged in
    if "logged_in" in st.session_state and st.session_state.logged_in:
        st.success(f"You are logged in as {st.session_state.username}")
        if st.button("Logout"):
            # Clear the session state
            for key in ["logged_in", "username"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return True
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        # Create a form for login
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                elif authenticate_user(username, password):
                    # Set session state
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    # Rerun the app to refresh the page
                    st.rerun()
                    return True
                else:
                    st.error("Invalid username or password")
    
    with register_tab:
        # Create a form for registration
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if not new_username or not new_password:
                    st.error("Please enter both username and password")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif user_exists(new_username):
                    st.error("Username already exists")
                else:
                    if register_user(new_username, new_password):
                        st.success(f"Registration successful for {new_username}! Please log in.")
                    else:
                        st.error("Registration failed")
    
    # Guest mode option
    st.divider()
    st.write("Don't want to create an account?")
    if st.button("Continue as Guest"):
        st.session_state.logged_in = True
        st.session_state.username = "guest"
        st.rerun()
        return True
        
    return False

def get_current_user():
    """Get the currently logged in user"""
    if "logged_in" in st.session_state and st.session_state.logged_in:
        return st.session_state.username
    return None