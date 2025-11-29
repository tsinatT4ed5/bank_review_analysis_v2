{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "aa0e6f42-f47c-4c9d-94db-e7c908730a0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c2ead66d-b2d1-4241-b95a-35123f6a2280",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Data:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>rating</th>\n",
       "      <th>date</th>\n",
       "      <th>bank</th>\n",
       "      <th>source</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>This application is very important and advanta...</td>\n",
       "      <td>5</td>\n",
       "      <td>11/27/2025</td>\n",
       "      <td>CBE</td>\n",
       "      <td>Google Play Store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>why didn't work this app?</td>\n",
       "      <td>1</td>\n",
       "      <td>11/27/2025</td>\n",
       "      <td>CBE</td>\n",
       "      <td>Google Play Store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>The app makes our life easier. Thank you CBE!</td>\n",
       "      <td>5</td>\n",
       "      <td>11/27/2025</td>\n",
       "      <td>CBE</td>\n",
       "      <td>Google Play Store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>this app very bad 👎</td>\n",
       "      <td>1</td>\n",
       "      <td>11/27/2025</td>\n",
       "      <td>CBE</td>\n",
       "      <td>Google Play Store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>the most advanced app. but how to stay safe?</td>\n",
       "      <td>5</td>\n",
       "      <td>11/27/2025</td>\n",
       "      <td>CBE</td>\n",
       "      <td>Google Play Store</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  rating        date bank  \\\n",
       "0  This application is very important and advanta...       5  11/27/2025  CBE   \n",
       "1                          why didn't work this app?       1  11/27/2025  CBE   \n",
       "2      The app makes our life easier. Thank you CBE!       5  11/27/2025  CBE   \n",
       "3                                this app very bad 👎       1  11/27/2025  CBE   \n",
       "4       the most advanced app. but how to stay safe?       5  11/27/2025  CBE   \n",
       "\n",
       "              source  \n",
       "0  Google Play Store  \n",
       "1  Google Play Store  \n",
       "2  Google Play Store  \n",
       "3  Google Play Store  \n",
       "4  Google Play Store  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Load the data using your function\n",
    "df = pd.read_csv(\"clean_bank_reviews (2).csv\")\n",
    "\n",
    "# Display first few rows\n",
    "print(\"Original Data:\")\n",
    "display(df.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "d1c0e355-e5b2-43c7-b1ce-f488c3ff79d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data shape after removing duplicates: (1187, 5)\n"
     ]
    }
   ],
   "source": [
    "# 1. Remove duplicates\n",
    "df = df.drop_duplicates()\n",
    "print(f\"Data shape after removing duplicates: {df.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "ff3c677f-0210-4ae8-9d5d-6612ed28c359",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1187, 5)\n"
     ]
    }
   ],
   "source": [
    "# 2. Handle missing values\n",
    "df = df.dropna()\n",
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "0eed39da-4626-4e62-ad68-7d6648ad88ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Normalize / clean text columns\n",
    "def clean_text(text):\n",
    "    if isinstance(text, str):\n",
    "        text = text.lower()  # lowercase\n",
    "        text = text.strip()  # remove leading/trailing spaces\n",
    "        text = re.sub(r'[^a-z0-9\\s]', '', text)  # remove special characters\n",
    "        text = re.sub(r'\\s+', ' ', text)  # replace multiple spaces with single space\n",
    "        return text\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "97af1fab-8671-493d-8d11-07e8033eb71d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply cleaning to all object (text) columns\n",
    "for col in df.select_dtypes(include=['object']).columns:\n",
    "    df[col] = df[col].apply(clean_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "4905c400-7e0f-40b9-b929-104d04d1a091",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Clean column names\n",
    "df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "209d661e-4b75-4083-8eb8-a09a746fe77f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cleaned Data:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>rating</th>\n",
       "      <th>date</th>\n",
       "      <th>bank</th>\n",
       "      <th>source</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>this application is very important and advanta...</td>\n",
       "      <td>5</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>why didnt work this app</td>\n",
       "      <td>1</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>the app makes our life easier thank you cbe</td>\n",
       "      <td>5</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>this app very bad</td>\n",
       "      <td>1</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>the most advanced app but how to stay safe</td>\n",
       "      <td>5</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  rating      date bank  \\\n",
       "0  this application is very important and advanta...       5  11272025  cbe   \n",
       "1                            why didnt work this app       1  11272025  cbe   \n",
       "2        the app makes our life easier thank you cbe       5  11272025  cbe   \n",
       "3                                 this app very bad        1  11272025  cbe   \n",
       "4         the most advanced app but how to stay safe       5  11272025  cbe   \n",
       "\n",
       "              source  \n",
       "0  google play store  \n",
       "1  google play store  \n",
       "2  google play store  \n",
       "3  google play store  \n",
       "4  google play store  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Show cleaned data\n",
    "print(\"Cleaned Data:\")\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "c64c3f62-507f-4d09-ae79-a20a36c1aa4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from dateutil import parser\n",
    "import re\n",
    "\n",
    "# Normalize text and dates\n",
    "def clean_text(text):\n",
    "    if isinstance(text, str):\n",
    "        text = text.strip()  # remove leading/trailing spaces\n",
    "        \n",
    "        # Try to parse as date\n",
    "        try:\n",
    "            parsed_date = parser.parse(text, fuzzy=False)\n",
    "            return parsed_date.strftime('%Y-%m-%d')  # normalize to YYYY-MM-DD\n",
    "        except (ValueError, OverflowError):\n",
    "            pass  # not a date, continue cleaning as text\n",
    "        \n",
    "        # Clean as regular text\n",
    "        text = text.lower()  # lowercase\n",
    "        text = re.sub(r'[^a-z0-9\\s]', '', text)  # remove special characters\n",
    "        text = re.sub(r'\\s+', ' ', text)  # replace multiple spaces with single space\n",
    "        return text\n",
    "    \n",
    "    return text\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "ff2f113e-523f-4358-a665-2aac65146d31",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>rating</th>\n",
       "      <th>date</th>\n",
       "      <th>bank</th>\n",
       "      <th>source</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>this application is very important and advanta...</td>\n",
       "      <td>5</td>\n",
       "      <td>2025-11-27</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>why didnt work this app</td>\n",
       "      <td>1</td>\n",
       "      <td>2025-11-27</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>the app makes our life easier thank you cbe</td>\n",
       "      <td>5</td>\n",
       "      <td>2025-11-27</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>this app very bad</td>\n",
       "      <td>1</td>\n",
       "      <td>2025-11-27</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>the most advanced app but how to stay safe</td>\n",
       "      <td>5</td>\n",
       "      <td>2025-11-27</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  rating        date bank  \\\n",
       "0  this application is very important and advanta...       5  2025-11-27  cbe   \n",
       "1                            why didnt work this app       1  2025-11-27  cbe   \n",
       "2        the app makes our life easier thank you cbe       5  2025-11-27  cbe   \n",
       "3                                 this app very bad        1  2025-11-27  cbe   \n",
       "4         the most advanced app but how to stay safe       5  2025-11-27  cbe   \n",
       "\n",
       "              source  \n",
       "0  google play store  \n",
       "1  google play store  \n",
       "2  google play store  \n",
       "3  google play store  \n",
       "4  google play store  "
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Ensure the date column is string\n",
    "df['date'] = df['date'].astype(str)\n",
    "\n",
    "# Function to convert MMDDYYYY to YYYY-MM-DD\n",
    "def normalize_date(date_str):\n",
    "    try:\n",
    "        # Parse date assuming MMDDYYYY format\n",
    "        month = int(date_str[:2])\n",
    "        day = int(date_str[2:4])\n",
    "        year = int(date_str[4:])\n",
    "        return f\"{year:04d}-{month:02d}-{day:02d}\"\n",
    "    except:\n",
    "        return np.nan  # if format is wrong, mark as missing\n",
    "\n",
    "# Apply normalization\n",
    "df['date'] = df['date'].apply(normalize_date)\n",
    "\n",
    "# Check result\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "e0aee1d9-986a-4e46-af12-b7594ee3878d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cleaned Data:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>rating</th>\n",
       "      <th>date</th>\n",
       "      <th>bank</th>\n",
       "      <th>source</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>this application is very important and advanta...</td>\n",
       "      <td>5</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>why didnt work this app</td>\n",
       "      <td>1</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>the app makes our life easier thank you cbe</td>\n",
       "      <td>5</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>this app very bad</td>\n",
       "      <td>1</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>the most advanced app but how to stay safe</td>\n",
       "      <td>5</td>\n",
       "      <td>11272025</td>\n",
       "      <td>cbe</td>\n",
       "      <td>google play store</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  rating      date bank  \\\n",
       "0  this application is very important and advanta...       5  11272025  cbe   \n",
       "1                            why didnt work this app       1  11272025  cbe   \n",
       "2        the app makes our life easier thank you cbe       5  11272025  cbe   \n",
       "3                                 this app very bad        1  11272025  cbe   \n",
       "4         the most advanced app but how to stay safe       5  11272025  cbe   \n",
       "\n",
       "              source  \n",
       "0  google play store  \n",
       "1  google play store  \n",
       "2  google play store  \n",
       "3  google play store  \n",
       "4  google play store  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Show cleaned data\n",
    "print(\"Cleaned Data:\")\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "5902a00b-fe90-4675-b6c2-2abfef92e78c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "File saved successfully!\n"
     ]
    }
   ],
   "source": [
    "# Save the cleaned DataFrame to a CSV file\n",
    "df.to_csv('cleaned_bank_reviews.csv', index=False)\n",
    "\n",
    "print(\"File saved successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "252d7238-8f61-4b4c-b987-8d21ecf85427",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
