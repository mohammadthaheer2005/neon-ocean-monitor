from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime

class MissionReportGenerator:
    """
    Generates a 'Scientific/Military' Grade PDF Report.
    V5.0 Update: Includes Light Intensity and Expanded Data.
    """
    def generate_pdf(self, location, inputs, result):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24, spaceAfter=20, textColor=colors.darkblue)
        sub_style = ParagraphStyle('Sub', parent=styles['Heading2'], textColor=colors.grey, fontSize=12, alignment=TA_CENTER)
        
        # 1. Header
        elements.append(Paragraph("NEON OCEAN: FINAL PROJECT REPORT", title_style))
        elements.append(Paragraph("OPTIMIZED HYBRID PREDICTIVE ANALYTICS MODEL", sub_style))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"<b>TARGET SECTOR:</b> {location.upper()} | <b>DATE:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # 2. Executive Summary
        risk_color = colors.red if "CRITICAL" in result['risk'] else (colors.orange if "Moderate" in result['risk'] else colors.green)
        
        data_summary = [
            ["PREDICTED GROWTH STATUS", result['risk']],
            ["CONFIDENCE PROBABILITY", f"{result['confidence']:.1f}%"],
            ["DETECTION ZONE", f"CLUSTER {result['cluster']} (Hybrid ML)"]
        ]
        
        t_sum = Table(data_summary, colWidths=[200, 250])
        t_sum.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.black),
            ('TEXTCOLOR', (0,0), (0,-1), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1,0), (1,0), risk_color), 
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('PADDING', (0,0), (-1,-1), 10),
        ]))
        elements.append(t_sum)
        elements.append(Spacer(1, 20))

        # 3. Comprehensive Sensor Data (Including Light)
        elements.append(Paragraph("COMPREHENSIVE SENSOR DATA", styles['Heading2']))
        
        sensor_data = [
            ["PARAMETER", "VALUE", "CONDITION"],
            ["Sea Surface Temp", f"{inputs['temperature']:.2f} °C", "NOAA LIVE FEED"],
            ["Light Intensity", f"{inputs.get('light_intensity', 'N/A')} PAR", "PHOTOSYNTHESIS DRIVER"],
            ["Nitrate (NO3)", f"{inputs['nitrate']:.2f} mg/L", "NUTRIENT LOAD"],
            ["Phosphate (PO4)", f"{inputs['phosphate']:.2f} mg/L", "NUTRIENT LOAD"],
            ["pH Level", f"{inputs['pH']:.1f}", "ACIDITY"],
            ["Turbidity", f"{inputs['turbidity']:.1f} NTU", "WATER CLARITY"]
        ]
        
        t_sens = Table(sensor_data, colWidths=[150, 150, 150])
        t_sens.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), # Header
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        elements.append(t_sens)
        elements.append(Spacer(1, 20))

        # 4. AI ROOT CAUSE DIAGNOSIS
        elements.append(Paragraph("AI ROOT CAUSE DIAGNOSIS", styles['Heading2']))
        
        # logic-based diagnosis
        nitrate = inputs['nitrate']
        phosphate = inputs['phosphate']
        oxygen = inputs.get('dissolved_oxygen', 6.5)
        
        cause = "Nutrient imbalances from coastal runoff"
        if nitrate > 5 and phosphate > 2: cause = "SYNERGISTIC EUTROPHICATION (Extreme N/P load)"
        elif nitrate > 5: cause = "NITROGEN-DRIVEN OVERGROWTH (Agricultural runoff)"
        elif phosphate > 2: cause = "PHOSPHATE SPIKE (Detergent/Industrial loading)"
        
        diagnosis_text = f"""
        <b>Primary Driver:</b> {cause}<br/>
        <b>Biomass Impact:</b> The current biomass density of {inputs.get('biomass', 'N/A')} mg/m³ is driven by high solar radiation and stagnant currents.
        """
        elements.append(Paragraph(diagnosis_text, styles['BodyText']))
        elements.append(Spacer(1, 15))

        # 5. MITIGATION & CHEMICAL PROTOCOLS
        elements.append(Paragraph("CHEMICAL MITIGATION PROTOCOLS", styles['Heading2']))
        
        chemical_data = [
            ["CHEMICAL AGENT", "DOSAGE GUIDE", "FUNCTION"],
            ["Alum (Al2(SO4)3)", "5 - 20 mg/L", "Phosphate Coagulation"],
            ["Modified Clay", "5 - 10 g/m²", "Flocculation / Cell Settling"],
            ["Phoslock (LMB)", "Based on P-Load", "Permanent P-Binding"],
            ["Hydrogen Peroxide", "1 - 2 mg/L", "Selective Cyanobacteria Control"]
        ]
        
        t_chem = Table(chemical_data, colWidths=[150, 150, 150])
        t_chem.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 10),
        ]))
        elements.append(t_chem)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
