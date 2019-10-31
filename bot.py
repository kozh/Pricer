from bottle import run, post

@post('/')  # our python function based endpoint
def main():  

	data = bottle_request.json  # <--- extract all request data
	print(data)

	return

if __name__ == '__main__':  
	run(host='localhost', port=8080, debug=True)