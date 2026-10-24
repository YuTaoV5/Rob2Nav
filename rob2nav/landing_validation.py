"""Offline independent segmentation of measured physics, without phase counters.

Checks motion only. Motor/target constraints, endpoints and video remain required.
"""
import math


def audit_landing_cycles(rows, dt, settings, expected_samples):
    if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid timestep')
    c=settings
    required=('cycles','entry_seconds','max_rebound_seconds','takeoff_confirm_seconds',
        'landing_confirm_seconds','min_airtime_seconds','min_rotation','max_rotation',
        'min_axis_dominance','landing_max_seconds','min_air_rotation','inverted_cos','landing_upright_cos')
    if any(k not in c for k in required):raise ValueError('Complete measured-motion settings required')
    if type(c['cycles']) is not int or c['cycles'] not in (2,3):raise ValueError('Invalid cycle count')
    if not all(type(c[k]) in (int,float) and math.isfinite(c[k]) for k in required):
        raise ValueError('Finite numeric thresholds required')
    if (not 5.5<=c['min_rotation']<c['max_rotation']<=7.1 or
        not 0<c['takeoff_confirm_seconds']<=c['min_airtime_seconds'] or
        not 0<c['landing_confirm_seconds']<c['landing_max_seconds']<=c['max_rebound_seconds'] or
        not 0<c['min_air_rotation']<=c['min_rotation'] or
        not 0<c['min_axis_dominance']<=1 or
        not -1<=c['inverted_cos']<0<c['landing_upright_cos']<=1):
        raise ValueError('Invalid or non-full-flip acceptance thresholds')
    finite=True;coverage=len(rows)==expected_samples and bool(rows)
    body=False;pitch=[0.];travel=[0.];segments=[];start=None
    for i,r in enumerate(rows):
        vals=[r['time'],r['upright'],*r['omega_body']]
        if len(r['omega_body'])!=3 or not all(math.isfinite(x) for x in vals):finite=False;break
        if type(r['foot_contact']) is not bool or type(r['body_contact']) is not bool:raise ValueError('Boolean contact required')
        coverage &= abs(r['time']-(i+1)*dt)<=max(1e-7,dt*1e-4)
        finite &= abs(r['upright'])<=1.00001
        body |= r['body_contact']
        pitch.append(pitch[-1]-r['omega_body'][1]*dt)
        travel.append(travel[-1]+math.sqrt(sum(x*x for x in r['omega_body']))*dt)
        if not r['foot_contact'] and start is None:start=i
        if r['foot_contact'] and start is not None:
            if (i-start)*dt+1e-7>=c['takeoff_confirm_seconds']:segments.append((start,i))
            start=None
    unfinished=start is not None and (len(rows)-start)*dt+1e-7>=c['takeoff_confirm_seconds']
    flights=[];connections=[];begin=next((i for i,r in enumerate(rows) if r['time']+1e-7>=c['entry_seconds']),len(rows))
    previous_landing=None
    if finite:
        for index,(a,b) in enumerate(segments):
            air_angle=pitch[b]-pitch[a];air_travel=travel[b]-travel[a]
            air_ok=((b-a)*dt+1e-7>=c['min_airtime_seconds'] and air_angle>=c['min_air_rotation'] and
                air_angle/max(air_travel,1e-12)>=c['min_axis_dominance'] and
                any(r['upright']<=c['inverted_cos'] for r in rows[a:b]))
            connection=None
            if previous_landing is not None:
                elapsed=(a-previous_landing)*dt+math.ceil(c['takeoff_confirm_seconds']/dt-1e-9)*dt
                connection=dict(seconds=elapsed,passed=elapsed<=c['max_rebound_seconds']+1e-7)
                connections.append(connection)
            stop=segments[index+1][0] if index+1<len(segments) else len(rows)
            consecutive=0;completion=None;total=None;dominance=None;overshot=False
            # Search raw landing samples; never consult execution phase or completed count.
            for j in range(b,stop):
                if (j-b+1)*dt>c['landing_max_seconds']+1e-7:break
                consecutive=consecutive+1 if rows[j]['foot_contact'] else 0
                if begin>a:break
                total=pitch[j+1]-pitch[begin];den=travel[j+1]-travel[begin]
                dominance=total/max(den,1e-12)
                overshot |= total>c['max_rotation']
                if (air_ok and not overshot and c['min_rotation']<=total<=c['max_rotation'] and
                    dominance>=c['min_axis_dominance'] and consecutive*dt+1e-7>=c['landing_confirm_seconds'] and
                    rows[j]['upright']>=c['landing_upright_cos']):
                    completion=j;break
            flights.append(dict(takeoff_s=rows[a]['time'],first_contact_s=rows[b]['time'],
                airtime_s=(b-a)*dt,air_rotation_rad=air_angle,air_evidence_pass=bool(air_ok),
                completion_s=None if completion is None else rows[completion]['time'],
                full_rotation_rad=total,axis_dominance=dominance,motion_pass=completion is not None))
            begin=(completion+1) if completion is not None else len(rows)
            previous_landing=b
    passed=bool(coverage and finite and not body and not unfinished and len(flights)==c['cycles'] and
        all(x['motion_pass'] for x in flights) and all(x['passed'] for x in connections))
    return dict(motion_pass=passed,scope='independent raw contact/orientation/rotation segmentation',
        coverage_complete=bool(coverage and finite),samples=len(rows),expected_samples=expected_samples,
        observed_flights=len(flights),expected_cycles=c['cycles'],flights=flights,connections=connections,
        unexpected_body_contact=body,unfinished_flight=unfinished,hardware_qualified=False)
