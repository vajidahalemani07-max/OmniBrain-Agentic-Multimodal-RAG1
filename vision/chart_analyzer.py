def extract_charts(page_result):
    #Extract chart elements from page analysis result.
    charts=[]
    if not isinstance(page_result,list):
        return charts
    for element in page_result:
        if isinstance(element,dict) and element.get("type")=="CHART":
            charts.append(element)
    return charts
    