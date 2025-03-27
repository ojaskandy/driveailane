from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    """Simple home page for testing deployment"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DriveSafe - Basic Test Page</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                text-align: center;
            }
            h1 {
                color: #333;
            }
            .message {
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f8f8;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <h1>DriveSafe Application</h1>
        <div class="message">
            <p>Basic deployment is working! The full app is being loaded.</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True) 