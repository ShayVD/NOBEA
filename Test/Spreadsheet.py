import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pprint


class Spreadsheet(object):

    def __init__(self):
        self.client = self.google_drive_login()
        self.sheet = self.client.open("extended_abstract").sheet1

    @staticmethod
    def google_drive_login():
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('../client_secret.json', scope)
        client = gspread.authorize(creds)
        return client

    def set_cell(self, row, column, data):
        self.sheet.update_cell(row, column, data)

    def get_cell(self, row, column):
        return self.sheet.cell(row, column)

    def set_row_string(self, row, string):
        lst = string.split(" ")
        for i in range(len(lst)):
            self.set_cell(row+1, i+1, lst[i])

    def get_row(self, row):
        return self.sheet.row_values(row)

    def clear_sheet(self):
        self.sheet.clear()


if __name__ == "__main__":
    # use creds to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('../client_secret.json', scope)
    client = gspread.authorize(creds)

    pp = pprint.PrettyPrinter()

    # row = ["I'm", "updating", "a", "spreadsheet", "from", "Python!"]
    # index = 3
    # sheet.delete_row(3)

    # result = sheet.get_all_records()
    # result = sheet.row_values(6)
    # result = sheet.col_values(6)
    # result = sheet.cell(6, 11)

    # pp.pprint(result)

    # sheet.update_cell(6, 11, "555-867-5309")
    # result = sheet.cell(6, 11)
    # pp.pprint(result)

    sheet = client.open("NOBEA_Results").sheet1
    sheet.update_cell(1, 1, "Hello World!")
    result = sheet.cell(1, 1)
    pp.pprint(result)


