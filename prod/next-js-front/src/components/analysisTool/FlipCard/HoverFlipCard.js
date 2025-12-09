'use client';

import React, { useState } from "react";
import ReactCardFlip from "react-card-flip";

const HoverFlipCard = ({ frontContent, backContent, color }) => {
  const [isFlipped, setIsFlipped] = useState(false);

  const handleMouseEnter = () => setIsFlipped(true);
  const handleMouseLeave = () => setIsFlipped(false);

  const [cardWidth, cardHeight] = [300, 170]; // Dimensions of the card
  const cardStyles = {
    front: {
      width: `${cardWidth}px`,
      height: `${cardHeight}px`,
      borderRadius: "0.5rem", // Rounded corners
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)", // Shadow
      backgroundColor: color, // Dynamic color for the front side
    },
    back: {
      width: `${cardWidth}px`,
      height: `${cardHeight}px`,
      borderRadius: "0.5rem",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
      backgroundColor: "#1a202c", // Default back color (dark gray)
    },
  };

  return (
    <div
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className="cursor-pointer"
    >
      <ReactCardFlip isFlipped={isFlipped} flipDirection="horizontal" cardStyles={cardStyles}>
        {/* Front Side */}
        <div className="w-full h-full flex flex-col justify-center items-center text-white p-4 rounded-lg shadow-lg">
          {frontContent}
        </div>

        {/* Back Side */}
        <div className="w-full h-full flex flex-col justify-center items-center text-white p-4 rounded-lg shadow-lg">
          {backContent}
        </div>
      </ReactCardFlip>
    </div>
  );
};

export default HoverFlipCard;
