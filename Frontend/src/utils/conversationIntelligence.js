/**
 * DisasterMind Conversational Intelligence Engine
 * 
 * Determines investigation workflows based on natural language query,
 * conversation history, and uploaded media. Coordinates backend APIs:
 * - /api/v1/intelligence/analyze-upload (Multimodal scene & internet intelligence)
 * - /api/v1/investigate/route (OSRM routing corridor, distance & hazard analysis)
 * - /api/v1/investigate/location (USGS, GDACS, Open-Meteo multi-hazard evidence)
 * - /api/v1/geocode/search (Nominatim coordinate verification)
 */

export function parseQueryIntent(query, context = {}) {
  const q = (query || '').trim();
  const lower = q.toLowerCase();

  // 1. Detect Location Correction
  const correctionMatch = q.match(/(?:the\s+)?location\s+is\s+(?:actually\s+)?([A-Za-z0-9\s,\.\-]+?)(?:\.|\?|$)/i)
    || q.match(/(?:this\s+is\s+(?:actually\s+)?(?:in\s+)?)([A-Za-z0-9\s,\.\-]+?)(?:\.|\?|$)/i)
    || q.match(/(?:it'?s\s+(?:actually\s+)?(?:in\s+)?)([A-Za-z0-9\s,\.\-]+?)(?:\.|\?|$)/i);
  
  const isCorrection = Boolean(correctionMatch && !lower.includes('how far') && !lower.includes('route'));
  const correctedLocation = isCorrection ? correctionMatch[1].trim() : null;

  // 2. Detect Date / Time Frame
  let targetDate = 'today';
  let dateMentioned = false;
  if (lower.includes('tomorrow')) {
    targetDate = 'tomorrow';
    dateMentioned = true;
  } else if (lower.includes('next weekend') || lower.includes('weekend')) {
    targetDate = 'next weekend';
    dateMentioned = true;
  } else if (lower.includes('in 3 days') || lower.includes('in three days')) {
    targetDate = 'in 3 days';
    dateMentioned = true;
  } else {
    const isoDateMatch = q.match(/\b(202\d-\d{2}-\d{2})\b/);
    if (isoDateMatch) {
      targetDate = isoDateMatch[1];
      dateMentioned = true;
    }
  }

  // 3. Detect Route / Transit / Distance Intent
  const routeKeywords = [
    'route', 'distance', 'how far', 'can i visit', 'visit this place',
    'directions', 'road conditions', 'highway', 'drive from', 'trip to',
    'travel to', 'transit corridor', 'closures along'
  ];
  const isRouteIntent = routeKeywords.some((kw) => lower.includes(kw));

  // Extract Origin
  let origin = null;
  const originFromMatch = q.match(/from\s+([A-Za-z0-9\s,\.\-]+?)(?:\s+to|\s+for|\s+on|\s+next|\s+tomorrow|\s+today|\?|$)/i);
  const distanceHowFarMatch = q.match(/how far\s+(?:is\s+)?(?:this(?:\s+place|\s+location)?|here)\s+from\s+([A-Za-z0-9\s,\.\-]+?)(?:\?|$)/i);
  const distanceCityCityMatch = q.match(/how far\s+(?:is\s+)?([A-Za-z0-9\s,\.\-]+?)\s+from\s+([A-Za-z0-9\s,\.\-]+?)(?:\?|$)/i);
  const routeBetweenMatch = q.match(/route\s+(?:between\s+)?([A-Za-z0-9\s,\.\-]+?)\s+(?:and|to)\s+([A-Za-z0-9\s,\.\-]+?)(?:\?|$)/i);

  if (distanceHowFarMatch) {
    origin = distanceHowFarMatch[1].trim();
  } else if (distanceCityCityMatch) {
    origin = distanceCityCityMatch[2].trim();
  } else if (routeBetweenMatch) {
    origin = routeBetweenMatch[1].trim();
  } else if (originFromMatch) {
    origin = originFromMatch[1].trim();
  } else if (context.route_context?.origin) {
    origin = context.route_context.origin;
  }

  // Clean origin from trailing words
  if (origin) {
    origin = origin.replace(/\s+(and|or|how|what|is|the)\s*$/i, '').trim();
  }

  // Extract Destination
  let destination = null;
  const destinationToMatch = q.match(/to\s+([A-Za-z0-9\s,\.\-]+?)(?:\s+from|\s+for|\s+on|\s+next|\s+tomorrow|\s+today|\?|$)/i);
  const visitCityMatch = q.match(/visit\s+([A-Za-z0-9\s,\.\-]+?)(?:\s+from|\s+for|\s+on|\s+next|\s+tomorrow|\s+today|\?|$)/i);

  if (distanceCityCityMatch) {
    destination = distanceCityCityMatch[1].trim();
  } else if (routeBetweenMatch) {
    destination = routeBetweenMatch[2].trim();
  } else if (destinationToMatch) {
    destination = destinationToMatch[1].trim();
  } else if (visitCityMatch && !lower.includes('this place') && !lower.includes('this location')) {
    destination = visitCityMatch[1].trim();
  }

  // If destination refers to current context
  const refersToCurrentLocation = lower.includes('this place') || lower.includes('this location') ||
    lower.includes('here') || lower.includes('the area') || lower.includes('this area');

  if ((!destination || refersToCurrentLocation) && context.confirmed_location) {
    destination = context.confirmed_location;
  }

  // 4. Detect Location Condition / Safety Intent
  const locationSafetyKeywords = [
    'safe today', 'is it safe', 'current conditions', 'what about tomorrow',
    'weather in', 'hazards in', 'situation in', 'earthquake in', 'floods in',
    'damage in', 'how is the environment'
  ];
  const isLocationSafetyIntent = locationSafetyKeywords.some((kw) => lower.includes(kw)) ||
    (dateMentioned && !isRouteIntent);

  // If location safety requested and no destination found, check for named location
  let locationTarget = destination || context.confirmed_location;
  if (!locationTarget) {
    const inLocationMatch = q.match(/(?:in|at|for)\s+([A-Za-z0-9\s,\.\-]+?)(?:\s+today|\s+tomorrow|\s+next|\?|$)/i);
    if (inLocationMatch) {
      locationTarget = inLocationMatch[1].trim();
    }
  }

  return {
    rawQuery: q,
    isCorrection,
    correctedLocation,
    isRouteIntent,
    origin,
    destination: destination || locationTarget,
    targetDate,
    dateMentioned,
    isLocationSafetyIntent,
    locationTarget,
    refersToCurrentLocation,
  };
}

/**
 * Execute unified conversational inquiry across backend services.
 */
export async function executeConversationalInquiry({
  question,
  files = [],
  context = {},
  searchLive = true,
  engineMode = 'deep_vlm',
  signal,
}) {
  const intent = parseQueryIntent(question, context);
  const results = {
    userQuery: question,
    intent,
    vlmData: null,
    routeData: null,
    locationData: null,
    establishedLocation: context.confirmed_location || null,
    establishedCoords: context.confirmed_coords || null,
    updatedRouteContext: context.route_context || null,
    error: null,
  };

  // Case 0: Location Correction
  if (intent.isCorrection && intent.correctedLocation) {
    results.establishedLocation = intent.correctedLocation;
    try {
      const locRes = await fetch('/api/v1/investigate/location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: intent.correctedLocation,
          target_date: intent.targetDate,
          question: `Assess current disaster hazards and conditions for ${intent.correctedLocation}.`,
        }),
        signal,
      });
      if (locRes.ok) {
        results.locationData = await locRes.json();
        if (results.locationData.location_coords) {
          results.establishedCoords = results.locationData.location_coords;
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') console.warn('Location re-investigation failed:', e);
    }
    return results;
  }

  // Case 1: Image(s) Attached
  if (files.length > 0 || (context.image_paths && context.image_paths.length > 0 && !intent.isRouteIntent && !intent.isLocationSafetyIntent)) {
    const formData = new FormData();
    formData.append('question', question || 'Analyze the uploaded disaster imagery.');
    formData.append('search_live', searchLive);
    formData.append('engine_mode', engineMode);

    if (context.historyText) {
      formData.append('conversation_history', context.historyText);
    }

    if (files.length > 0) {
      files.forEach((file) => formData.append('images', file));
    } else if (context.image_paths && context.image_paths.length > 0) {
      formData.append('existing_image_paths', JSON.stringify(context.image_paths));
    }

    const vlmRes = await fetch('/api/v1/intelligence/analyze-upload', {
      method: 'POST',
      body: formData,
      signal,
    });

    if (!vlmRes.ok) {
      const errJson = await vlmRes.json().catch(() => ({}));
      throw new Error(errJson.detail || `Vision analysis failed (${vlmRes.status})`);
    }

    results.vlmData = await vlmRes.json();

    // Check if location was identified
    const geo = results.vlmData?.geolocation_match;
    if (geo?.place_name) {
      results.establishedLocation = geo.place_name;
      if (geo.latitude && geo.longitude) {
        results.establishedCoords = { latitude: geo.latitude, longitude: geo.longitude };
      }
    }
  }

  // Determine active destination (from intent, vlm, or context)
  const activeDestination = intent.destination || results.establishedLocation || context.confirmed_location;

  // Case 2: Route Intent (Origin + Destination)
  if (intent.isRouteIntent && intent.origin && activeDestination) {
    try {
      const routeRes = await fetch('/api/v1/investigate/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin: intent.origin,
          destination: activeDestination,
          departure_time: intent.targetDate,
          question: question,
        }),
        signal,
      });

      if (routeRes.ok) {
        results.routeData = await routeRes.json();
        results.updatedRouteContext = {
          origin: intent.origin,
          destination: activeDestination,
          departure_time: intent.targetDate,
        };
        results.establishedLocation = activeDestination;
      } else {
        const errJson = await routeRes.json().catch(() => ({}));
        results.routeError = errJson.detail || 'Route corridor calculation unavailable.';
      }
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      console.warn('Route investigation error:', e);
      results.routeError = e.message;
    }
  }
  // Case 3: Route Intent but NO Origin (e.g. "Can I visit next weekend?") -> Check destination conditions
  else if (intent.isRouteIntent && !intent.origin && activeDestination) {
    try {
      const locRes = await fetch('/api/v1/investigate/location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: activeDestination,
          target_date: intent.targetDate,
          question: question,
        }),
        signal,
      });

      if (locRes.ok) {
        results.locationData = await locRes.json();
        results.establishedLocation = activeDestination;
      }
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      console.warn('Destination condition check error:', e);
    }
  }
  // Case 4: Location Hazard / Safety Intent
  else if (intent.isLocationSafetyIntent && activeDestination && !results.vlmData) {
    try {
      const locRes = await fetch('/api/v1/investigate/location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: activeDestination,
          target_date: intent.targetDate,
          question: question,
        }),
        signal,
      });

      if (locRes.ok) {
        results.locationData = await locRes.json();
        results.establishedLocation = activeDestination;
        if (results.locationData.location_coords) {
          results.establishedCoords = results.locationData.location_coords;
        }
      } else {
        const errJson = await locRes.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Location hazard assessment failed.');
      }
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      console.warn('Location investigation error:', e);
      // Fallback to text intelligence if location service failed
      if (!results.locationData) {
        const formData = new FormData();
        formData.append('question', question);
        formData.append('search_live', searchLive);
        formData.append('engine_mode', engineMode);
        if (context.historyText) formData.append('conversation_history', context.historyText);
        const fallbackRes = await fetch('/api/v1/intelligence/analyze-upload', {
          method: 'POST',
          body: formData,
          signal,
        });
        if (fallbackRes.ok) results.vlmData = await fallbackRes.json();
      }
    }
  }
  // Case 5: Text-only General Inquiry
  else if (!results.vlmData && !results.routeData && !results.locationData) {
    const formData = new FormData();
    formData.append('question', question);
    formData.append('search_live', searchLive);
    formData.append('engine_mode', engineMode);
    if (context.historyText) formData.append('conversation_history', context.historyText);
    if (context.image_paths && context.image_paths.length > 0) {
      formData.append('existing_image_paths', JSON.stringify(context.image_paths));
    }

    const res = await fetch('/api/v1/intelligence/analyze-upload', {
      method: 'POST',
      body: formData,
      signal,
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Analysis failed (${res.status})`);
    }

    results.vlmData = await res.json();
    const geo = results.vlmData?.geolocation_match;
    if (geo?.place_name) {
      results.establishedLocation = geo.place_name;
      if (geo.latitude && geo.longitude) {
        results.establishedCoords = { latitude: geo.latitude, longitude: geo.longitude };
      }
    }
  }

  return results;
}
